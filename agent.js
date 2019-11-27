
const tf = require('@tensorflow/tfjs');

const {createDeepQNetwork} = require('./dqn');
const {getRandomAction, SnakeGame, NUM_ACTIONS, ALL_ACTIONS, getStateTensor} = require('./game');
const ReplayMemory = require('./replay_memory');

module.exports = class TradingAgent {
  /**
   * Constructor of SnakeGameAgent.
   *
   * @param {SnakeGame} game A game object.
   * @param {object} config The configuration object with the following keys:
   *   - `replayBufferSize` {number} Size of the replay memory. Must be a
   *     positive integer.
   *   - `epsilonInit` {number} Initial value of epsilon (for the epsilon-
   *     greedy algorithm). Must be >= 0 and <= 1.
   *   - `epsilonFinal` {number} The final value of epsilon. Must be >= 0 and
   *     <= 1.
   *   - `epsilonDecayFrames` {number} The # of frames over which the value of
   *     `epsilon` decreases from `episloInit` to `epsilonFinal`, via a linear
   *     schedule.
   */
  constructor(game, config) {

    this.game = game;

    this.epsilonInit = config.epsilonInit;
    this.epsilonFinal = config.epsilonFinal;
    this.epsilonDecayFrames = config.epsilonDecayFrames;
    this.epsilonIncrement_ = (this.epsilonFinal - this.epsilonInit) / this.epsilonDecayFrames;

    this.onlineNetwork = createDeepQNetwork(NUM_ACTIONS);
    this.targetNetwork = createDeepQNetwork(NUM_ACTIONS);

    this.optimizer = tf.train.sgd(config.learningRate);

    this.replayBufferSize = config.replayBufferSize;
    this.replayMemory = new ReplayMemory(config.replayBufferSize);
    this.frameCount = 0;
    this.reset();
  }

  reset() {
    this.cumulativeReward_ = 0;
    this.tradesMade_ = 0;
    this.game.reset();
  }

  /**
   * Play one step of the game.
   *
   * @returns {number | null} If this step leads to the end of the game,
   *   the total reward from the game as a plain number. Else, `null`.
   */
  playStep() {
    this.epsilon = this.frameCount >= this.epsilonDecayFrames ?
        this.epsilonFinal :
        this.epsilonInit + this.epsilonIncrement_  * this.frameCount;
    this.frameCount++;

    // The epsilon-greedy algorithm.
    let action;
    const state = this.game.getState();
    if (Math.random() < this.epsilon) {
      // Pick an action at random.
      action = getRandomAction();
    } else {
      // Greedily pick an action based on online DQN output.
      tf.tidy(() => {
        const stateTensor = getStateTensor(state)
        const prediction = this.onlineNetwork.predict(stateTensor);
        // console.log(stateTensor.arraySync(), prediction.arraySync())
        action = ALL_ACTIONS[prediction.argMax(-1).dataSync()[0]];
      });
    }

    const {state: nextState, reward, done, tradeMade} = this.game.step(action);

    this.replayMemory.append([state, action, reward, done, nextState]);
    // console.log([state, action, reward, done, nextState])

    this.cumulativeReward_ += reward;
    if (tradeMade) {
      this.tradesMade_++;
    }
    const output = {
      action,
      cumulativeReward: this.cumulativeReward_,
      done,
      tradesMade: this.tradesMade_
    };
    if (done) {
      this.reset();
    }
    return output;
  }

  /**
   * Perform training on a randomly sampled batch from the replay buffer.
   *
   * @param {number} batchSize Batch size.
   * @param {number} gamma Reward discount rate. Must be >= 0 and <= 1.
   * @param {tf.train.Optimizer} optimizer The optimizer object used to update
   *   the weights of the online network.
   */
  trainOnReplayBatch(batchSize, gamma, optimizer) {
    // Get a batch of examples from the replay buffer.
    const batch = this.replayMemory.sample(batchSize);
    // console.log(batch[0])
    const lossFunction = () => tf.tidy(() => {
      const stateTensor = getStateTensor(batch.map(example => example[0]));
      const actionTensor = tf.tensor1d(batch.map(example => example[1]), 'int32');
      const qs = this.onlineNetwork.apply(stateTensor, {training: true}).mul(tf.oneHot(actionTensor, NUM_ACTIONS)).sum(-1);

      const rewardTensor = tf.tensor1d(batch.map(example => example[2]));
      const nextStateTensor = getStateTensor(batch.map(example => example[4]));
      const nextMaxQTensor = this.targetNetwork.predict(nextStateTensor).max(-1);
      const doneMask = tf.scalar(1).sub(tf.tensor1d(batch.map(example => example[3])).asType('float32'));
      const targetQs = rewardTensor.add(nextMaxQTensor.mul(doneMask).mul(gamma));
      // console.log('targetQs', targetQs.arraySync())
      // console.log('qs', qs.arraySync())
      return tf.losses.meanSquaredError(targetQs, qs);
    });

    // Calculate the gradients of the loss function with repsect to the weights
    // of the online DQN.
    const grads = tf.variableGrads(lossFunction);
    // Use the gradients to update the online DQN's weights.
    optimizer.applyGradients(grads.grads);
    tf.dispose(grads);
    // TODO(cais): Return the loss value here?
  }
}
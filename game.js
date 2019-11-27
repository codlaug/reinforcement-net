/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const tf = require('@tensorflow/tfjs');


// TODO(cais): Tune these parameters.
const NO_FRUIT_REWARD = -0.2;
const FRUIT_REWARD = 10;
const DEATH_REWARD = -10;


const ACTION_HOLD = 0;
const ACTION_BUY = 1;
const ACTION_SELL = 2;

const ALL_ACTIONS = [ACTION_HOLD, ACTION_BUY, ACTION_SELL];
const NUM_ACTIONS = ALL_ACTIONS.length;

module.exports.ALL_ACTIONS = ALL_ACTIONS
module.exports.NUM_ACTIONS = NUM_ACTIONS


function getRandomInteger(min, max) {
  // Note that we don't reuse the implementation in the more generic
  // `getRandomIntegers()` (plural) below, for performance optimization.
  return Math.floor((max - min) * Math.random()) + min;
}

/**
 * Generate a random action among all possible actions.
 *
 * @return {0 | 1 | 2} Action represented as a number.
 */
module.exports.getRandomAction = function getRandomAction() {
  return getRandomInteger(0, NUM_ACTIONS);
}




module.exports.TradingGame = class TradingGame {
  /**
   * Constructor of SnakeGame.
   *
   * @param {object} args Configurations for the game. Fields include:
   *   - height {number} height of the board (positive integer).
   *   - width {number} width of the board (positive integer).
   *   - numFruits {number} number of fruits present on the screen
   *     at any given step.
   *   - initLen {number} initial length of the snake.
   */
  constructor(args) {
    if (args == null) {
      args = {};
    }
    

    this.data = []

    for(let i = 0; i < 40; ++i) {
      this.data[i] = 10+Math.cos(i)
    }

    this.reset();
  }

  /**
   * Reset the state of the game.
   *
   * @return {object} Initial state of the game.
   *   See the documentation of `getState()` for details.
   */
  reset() {
    this.currentIndex = 0;
    this.assets = 0;
    this.currency = 50;
    this.startingCurrency = this.currency;
    return this.getState();
  }

  /**
   * Perform a step of the game.
   *
   * @param {0 | 1 | 2 | 3} action The action to take in the current step.
   *   The meaning of the possible values:
   *     0 - hold
   *     1 - buy
   *     2 - sell
   * @return {object} Object with the following keys:
   *   - `reward` {number} the reward value.
   *     - 0 if no fruit is eaten in this step
   *     - 1 if a fruit is eaten in this step
   *   - `state` New state of the game after the step.
   *   - `fruitEaten` {boolean} Whether a fruit is easten in this step.
   *   - `done` {boolean} whether the game has ended after this step.
   *     A game ends when the head of the snake goes off the board or goes
   *     over its own body.
   */
  step(action) {
    
    // Calculate the coordinates of the new head and check whether it has
    // gone off the board, in which case the game will end.
    let done = false;

    // Check if the head goes over the snake's body, in which case the
    // game will end.
    if(this.currentIndex >= this.data.length-2) {
        done = true;
    }
    

    if (done) {
      const endingBalance = (this.currency + (this.data[this.currentIndex-1] * this.assets)) - this.startingCurrency
      console.log('DONE~! REWARD:', endingBalance)
      return {reward: endingBalance === 0 ? -1 : endingBalance, done};
    }

    const assetPrice = this.data[this.currentIndex]



    // Check if a fruit is eaten.
    let reward = 0;
    
    
    if(action === ACTION_BUY) {
      if(this.currency > 0) {
        const currencyToSpend = Math.min(10, this.currency)
        this.assets += currencyToSpend / assetPrice
        this.currency -= currencyToSpend
        // console.log(`BUY @${assetPrice}`, `assets: ${this.assets}`, `currency: ${this.currency}`)
      }
    } else if(action === ACTION_SELL) {
      if(this.assets > 0) {
        const assetsToSell = this.assets > this.assets/5.0 ? this.assets/5.0 : this.assets;
        this.assets -= assetsToSell
        this.currency += assetsToSell * assetPrice
        // console.log(`SELL @${assetPrice}`, `assets: ${this.assets}`, `currency: ${this.currency}`)
      }
    }

    this.currentIndex += 1

    const state = this.getState();
    return {reward, state, done, tradeMade: action !== ACTION_HOLD};
  }



  /**
   * Get plain JavaScript representation of the game state.
   *
   * @return An object with two keys:
   *   - s: {Array<[number, number]>} representing the squares occupied by
   *        the snake. The array is ordered in such a way that the first
   *        element corresponds to the head of the snake and the last
   *        element corresponds to the tail.
   *   - f: {Array<[number, number]>} representing the squares occupied by
   *        the fruit(s).
   */
  getState() {
    return {
      assets: this.assets,
      currency: this.currency,
      price: this.data[this.currentIndex],
      nextPrice: this.data[this.currentIndex+1]
    }
  }
}

/**
 * Get the current state of the game as an image tensor.
 *
 * @param {object | object[]} state The state object as returned by
 *   `SnakeGame.getState()`, consisting of two keys: `s` for the snake and
 *   `f` for the fruit(s). Can also be an array of such state objects.
 * @param {number} h Height.
 * @param {number} w With.
 * @return {tf.Tensor} A tensor of shape [numExamples, height, width, 2] and
 *   dtype 'float32'
 *   - The first channel uses 0-1-2 values to mark the snake.
 *     - 0 means an empty square.
 *     - 1 means the body of the snake.
 *     - 2 means the head of the snake.
 *   - The second channel uses 0-1 values to mark the fruits.
 *   - `numExamples` is 1 if `state` argument is a single object or an
 *     array of a single object. Otherwise, it will be equal to the length
 *     of the state-object array.
 */

module.exports.getStateTensor = function getStateTensor(state) {
  if (!Array.isArray(state)) {
    state = [state];
  }
  const numExamples = state.length;
  // TODO(cais): Maintain only a single buffer for efficiency.
  const buffer = tf.buffer([numExamples, 4]);

  for (let n = 0; n < numExamples; ++n) {
    if (state[n] == null) {
      continue;
    }

    buffer.set(state[n].assets, n, 0);
    buffer.set(state[n].currency, n, 1);
    buffer.set(state[n].price, n, 2);
    buffer.set(state[n].nextPrice, n, 3);
  }
  return buffer.toTensor();
}
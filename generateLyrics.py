from __future__ import print_function
import numpy as np
import random
import tensorflow as tf
from six.moves import range
import pandas as pd
import string
import enchant
from random import randint




dfLyrics = pd.read_csv('.../DataSets/metroLyrics.csv')
sFile = ".../Songs/CAFElyrics.txt"
aArtists = ['coldplay',  'adele', 'frank-sinatra', 'ed-sheeran']  




def getLyrics(aArtists, dfLyrics):
    #Pulls the lyrics from each artist and returns it in an array
    aLyrics = []
    for sArtist in aArtists:
        aSongs = dfLyrics[dfLyrics["artist"] == sArtist]['lyrics']
        for sSong in aSongs:
            aLyrics.append(sSong)
        print(len(aLyrics))
    return aLyrics


def oneText(aLyrics):
    #Concatinates stings from the array aLyrics
    sLyrics = ""
    for sLyr in aLyrics:
        ## df contains nan values, must be deleted later
        ##For now, ignore non-strings
        if type(sLyr) == str:
            sLyrics += sLyr
    return sLyrics


def removePunt(sLyrics):
    #Removes puntuation and new lines
    sPunct = string.punctuation
    sLyrics = sLyrics.replace('\n', ' ')
    for sChar in sPunct:
        sLyrics = sLyrics.replace(sChar, "")
    return sLyrics

def singleSpace(sLyrics):
    #Return the string with only single spaces
    nSpaces = 0
    sNew = ''
    for sChar in sLyrics:
        if sChar != ' ':
            sNew += str(sChar)
            nSpaces = 0
        elif nSpaces == 0:
            sNew += ' '
            nSpaces += 1
    return sNew


def wordDes(sLyrics):
    ## Returns array of unique words from sLyrics
    aVocab = []
    sCopy = sLyrics[:]
    sWord = ''
    nPos = 0
    x = 0
    if sCopy[0] == " ":
        sCopy = sCopy[1:]


    while len(sCopy) > 0:
        step = sCopy.find(' ')
        sWord = sCopy[:step]

        aVocab.append(sWord)
        sCopy = sCopy[step + 1:]
        if sCopy[0:len(sWord)] == sWord:
            sCopy = sCopy[len(sWord)+1:]
        sWord = " " + sWord + " "

        sCopy = sCopy.replace(sWord, " ")
        sWord = ""
    return aVocab


def artistDict(aVocab):
    ##Creates enchant dictionary for list of words
    ##enchant.request_pwl_dict() technically tries to open a file
    ##Need to create function that just creates the dict alone
    dArtist = enchant.request_pwl_dict("_")
    for sWord in aVocab:
        dArtist.add(sWord)
    return dArtist

def AutoCorrect(sSong, dWords):
    ##Spell check on each word in sSong
    ##Return corrected string sCorrected
    if sSong[-1] == " ":
        sSong = sSong[:-1]
    nLen = len(sSong)
    sWord = ''
    sCorrected = ''
    for nPos in range(nLen):
        if (sSong[nPos] == " ") and sWord != "":
            aSuggestions = dWords.suggest(sWord)
            if len(aSuggestions) > 0:
                nSugg = len(aSuggestions) -1
                nRand = randint(0, nSugg)
                # sCorrected += aSuggestions[nRand] + " "
                sCorrected += aSuggestions[0] + " "
            sWord = ''
        else:
            sWord += sSong[nPos]
    if sWord != "":
        aSuggestions = dWords.suggest(sWord)
        if len(aSuggestions) > 0:
            nSugg = len(aSuggestions) - 1
            nRand = randint(0, nSugg)
            sCorrected += aSuggestions[nRand] + " "
    return sCorrected



def writeLyrics(sPath, aArray):
    #Create file with the new lyrics
    tFile = open(sPath, "w")
    nCount = 0
    sLine = ''
    for nLines in aArray:
        if nCount == 4:
            sLine = "\n"
        tFile.write(nLines + "\n" + sLine)
        sLine = ''
        nCount = (nCount + 1)%5
    tFile.close()
    return "OK"





aLyrics = getLyrics(aArtists, dfLyrics)
sLyrics = oneText(aLyrics)


print(len(sLyrics) , ".. That is a lot of letters!")

sLower = sLyrics.lower()

sLyrics = removePunt(sLower)
aVocab =  wordDes(sLyrics)
dWords =  artistDict(aVocab)

valid_size = 1000
valid_text = sLyrics[:valid_size]
train_text = sLyrics[valid_size:]
train_size = len(train_text)

vocabulary_size = len(string.ascii_lowercase) + 1
first_letter = ord(string.ascii_lowercase[0])

batch_size = 64
num_unrollings = 10


def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 1
    elif char == ' ':
        return 0
    else:
        # print('Unexpected character: %s' % char)
        return 0


def id2char(dictid):
    if dictid > 0:
        return chr(dictid + first_letter - 1)
    else:
        return ' '

class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, char2id(self._text[self._cursor[b]])] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches


def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(c) for c in np.argmax(probabilities, 1)]


def batches2string(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]
    return s


train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)


def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p

def random_distribution():
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
  return b/np.sum(b, 1)[:,None]

def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p

def random_distribution():
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
  return b/np.sum(b, 1)[:,None]


def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p

def random_distribution():
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
  return b/np.sum(b, 1)[:,None]


num_nodes = 64

graph = tf.Graph()
with graph.as_default():
    # Parameters:
    # Input gate: input, previous output, and bias.
    ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ib = tf.Variable(tf.zeros([1, num_nodes]))
    # Forget gate: input, previous output, and bias.
    fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    fb = tf.Variable(tf.zeros([1, num_nodes]))
    # Memory cell: input, state and bias.
    cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    cb = tf.Variable(tf.zeros([1, num_nodes]))
    # Output gate: input, previous output, and bias.
    ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ob = tf.Variable(tf.zeros([1, num_nodes]))
    # Variables saving state across unrollings.
    saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    # Classifier weights and biases.
    w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([vocabulary_size]))


    # Definition of the cell computation.
    def lstm_cell(i, o, state):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates."""
        input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
        forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
        update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
        state = forget_gate * state + input_gate * tf.tanh(update)
        output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
        return output_gate * tf.tanh(state), state


    # Input data.
    train_data = list()
    for _ in range(num_unrollings + 1):
        train_data.append(
            tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size]))
    train_inputs = train_data[:num_unrollings]
    train_labels = train_data[1:]  # labels are inputs shifted by one time step.

    # Unrolled LSTM loop.
    outputs = list()
    output = saved_output
    state = saved_state
    for i in train_inputs:
        output, state = lstm_cell(i, output, state)
        outputs.append(output)

    # State saving across unrollings.
    with tf.control_dependencies([saved_output.assign(output),
                                  saved_state.assign(state)]):
        # Classifier.
        logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), w, b)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.concat(train_labels, 0), logits=logits))

    # Optimizer.
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        10.0, global_step, 5000, 0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(
        zip(gradients, v), global_step=global_step)

    # Predictions.
    train_prediction = tf.nn.softmax(logits)

    # Sampling and validation eval: batch 1, no unrolling.
    sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
    saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
    saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
    reset_sample_state = tf.group(
        saved_sample_output.assign(tf.zeros([1, num_nodes])),
        saved_sample_state.assign(tf.zeros([1, num_nodes])))
    sample_output, sample_state = lstm_cell(
        sample_input, saved_sample_output, saved_sample_state)
    with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                  saved_sample_state.assign(sample_state)]):
        sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))



num_steps = 35000
summary_frequency = 100
aFinal = []

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batches = train_batches.next()
    feed_dict = dict()
    for i in range(num_unrollings + 1):
      feed_dict[train_data[i]] = batches[i]
    _, l, predictions, lr = session.run(
      [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print(
        'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
      labels = np.concatenate(list(batches)[1:])
      print('Minibatch perplexity: %.2f' % float(
        np.exp(logprob(predictions, labels))))
      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print('=' * 80)
        for _ in range(5):
          feed = sample(random_distribution())
          sentence = characters(feed)[0]
          reset_sample_state.run()
          for _ in range(79):
            prediction = sample_prediction.eval({sample_input: feed})
            feed = sample(prediction)
            sentence += characters(feed)[0]
          sSong = singleSpace(sentence)
          sCorr = AutoCorrect(sSong, dWords)
          aFinal.append(sCorr)
        print('=' * 80)
      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      for _ in range(valid_size):
        b = valid_batches.next()
        predictions = sample_prediction.eval({sample_input: b[0]})
        valid_logprob = valid_logprob + logprob(predictions, b[1])
      print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))


print(writeLyrics(sFile, aFinal))

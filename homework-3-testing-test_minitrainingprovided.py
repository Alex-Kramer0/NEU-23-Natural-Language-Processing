import unittest
import lm_model as lm_utils
from lm_model import LanguageModel

# Uses the following training files, in the training_files directory
# iamsam.txt
# iamsam2.txt
# unknowns_mixed.txt
# unknowns.txt

class TestMiniTraining(unittest.TestCase):
  
  
  def test_createunigrammodellaplace(self):
    unigram = LanguageModel(1)
    self.assertEqual(1, 1, msg="tests constructor for 1")

  def test_createbigrammodellaplace(self):
    bigram = LanguageModel(2)
    self.assertEqual(1, 1, msg="tests constructor for 2")

  def test_unigramlaplace(self):
    n = 1
    lm = LanguageModel(n)
    sentences = lm_utils.read_file("training_files/iamsam.txt")
    tokens = lm_utils.tokenize(sentences, n, by_char=False)
    lm.train(tokens)
    # ((2 + 1) / (10 + 5))
    self.assertAlmostEqual(.2, lm.score("<s>".split()), 
                           msg="tests probability of <s>, trained on iamsam.txt")
    # ((2 + 1) / (10 + 5)) ** 2
    self.assertAlmostEqual(.04, lm.score("<s> </s>".split()), 
                           msg="tests probability of <s> </s>, trained on iamsam.txt")
    # ((2 + 1) / (10 + 5)) ** 3
    self.assertAlmostEqual(.008, lm.score("<s> i </s>".split()), 
                            msg="tests probability of <s> i </s>, trained on iamsam.txt")

  def test_unigramunknownslaplace(self):
    n = 1
    lm = LanguageModel(n)
    sentences = lm_utils.read_file("training_files/unknowns_mixed.txt")
    tokens = lm_utils.tokenize(sentences, n, by_char=False)
    lm.train(tokens)
    # ((1 + 1) / (11 + 6))
    self.assertAlmostEqual(2 / 17, lm.score(["flamingo"]), places=3, 
                           msg="tests probability of flamingo, trained on unknowns_mixed.txt")

  def test_bigramunknownslaplace(self):
    n = 2
    lm = LanguageModel(n)
    sentences = lm_utils.read_file("training_files/unknowns_mixed.txt")
    tokens = lm_utils.tokenize(sentences, n, by_char=False)
    lm.train(tokens)
    # (0 + 1) / (2 + 6)
    self.assertAlmostEqual(1 / 8, lm.score("<s> flamingo".split()),
                            places=3,
                              msg="tests probability of <s> flamingo, trained on unknowns_mixed.txt")

  def test_bigramlaplace(self):
    n = 2
    lm = LanguageModel(n)
    sentences = lm_utils.read_file("training_files/iamsam2.txt")
    tokens = lm_utils.tokenize(sentences, n, by_char=False)
    lm.train(tokens)
    # (2 + 1) / (4 + 6)

    self.assertAlmostEqual(.3, lm.score(["<s>", "i"]), 
                           msg="tests probability of <s> i, trained on iamsam2.txt")
    # ((2 + 1) / (4 + 6)) * ((4 + 1) / (4 + 6)) * ((2 + 1) / (4 + 6))
    self.assertAlmostEqual(.045, lm.score("<s> i am </s>".split()),
                            msg="tests probability of <s> i am </s>, trained on iamsam2.txt")

  def test_generatebigramconcludes(self):
    n = 2
    lm = LanguageModel(n)
    sentences = lm_utils.read_file("training_files/iamsam2.txt")
    tokens = lm_utils.tokenize(sentences, n, by_char=False)
    lm.train(tokens)

    sents = lm.generate(2)
    self.assertEqual(2, len(sents), 
                     msg = "tests that you generated 2 sentences and that generate concluded\nDOES NOT TEST CORRECTNESS")
    print(sents)

  def test_generateunigramconcludes(self):
    n = 1
    lm = LanguageModel(n)
    sentences = lm_utils.read_file("training_files/iamsam2.txt")
    tokens = lm_utils.tokenize(sentences, n, by_char=False)
    lm.train(tokens)
    sents = lm.generate(2)
    self.assertEqual(2, len(sents), msg = "tests that you generated 2 sentences and that generate concluded\nDOES NOT TEST CORRECTNESS")
    print(sents)

  def test_onlyunknownsgenerationandscoring(self):
    n = 1
    lm = LanguageModel(1)
    sentences = lm_utils.read_file("training_files/unknowns.txt")
    tokens = lm_utils.tokenize(sentences, n, by_char=False)
    lm.train(tokens)

    # sentences should only contain unk tokens
    sents = lm.generate(5)
    for sent in sents:
      if len(sent) > 2:
        for word in sent[1:-1]:
          self.assertEqual("<UNK>", word.upper(), msg = "tests that all middle words in generated sentences are <UNK>, unigrams")

    # probability of unk should be v high
    score = lm.score(["porcupine"])
    # (6 + 1) / (10 + 3)
    self.assertAlmostEqual(.5385, score, places=3, msg = "tests probability of porcupine, trained on unknowns.txt, unigrams")

    # and then for bigrams
    n = 2
    lm = LanguageModel(n)
    sentences = lm_utils.read_file("training_files/unknowns.txt")
    tokens = lm_utils.tokenize(sentences, n, by_char=False)
    lm.train(tokens)

    # sentences should only contain unk tokens
    sents = lm.generate(5)
    for sent in sents:
      if len(sent) > 2:
        for word in sent[1:-1]:
          self.assertEqual("<UNK>", word.upper(), msg = "tests that all middle words in generated sentences are <UNK>, bigrams")

    # probability of unk should be v high
    score = lm.score(["porcupine", "wombat"])
    # (4 + 1) / (6 + 3)
    self.assertAlmostEqual(.5555555, score, places=3, msg = "tests probability of porcupine wombat, trained on unknowns.txt, bigrams")
    

if __name__ == "__main__":
  unittest.main()

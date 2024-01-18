#NOTE: the term "toxic" was defined by our group as sepcified in the document submission

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import emoji

print("STARTING\n" * 50)

v = DictVectorizer()

emoji_list = emoji.EMOJI_DATA.keys()

def replace_all(stringo, listo):
    output = stringo
    for thing in listo:
        output = output.replace(thing, " ")
    return output

def dictize_tweet(tweet):
    new_tweet = ""
    for c in tweet:
        if c in emoji_list:
            new_tweet += " " + c + " "
        else:
            new_tweet += c
    words = replace_all(new_tweet, ["\\n"] + list(" ,-.;:\"'“”")).lower().split()
    return {i:words.count(i) for i in set(words)}

def get_svector(word):
    with open("svectors200.txt", "r") as file:
        for line in file:
            pair = line.split(" ", 1)
            if pair[0] == word:
                return list(map(lambda thing: float(thing), pair[1].split()))
    return None

print(dictize_tweet("I, love ducks" + "\N{duck}" * 3))

with open("tweets.txt", "r") as file:
    lines = file.readlines()

with open("results.txt", "r") as file:
    desired = list(map(lambda thing: int(thing.strip()), file.readlines()))

#confusion matrix
def get_cm(reals, predicteds):
    cm = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for real, predicted in zip(reals, predicteds):
        if int(real):
            if int(predicted):
                cm["TP"] += 1
            else:
                cm["FN"] += 1
        else:
            if int(predicted):
                cm["FP"] += 1
            else:
                cm["TN"] += 1
    return cm

acc_total = 0
p_total = 0
r_total = 0

def new_predict(clf, data, threshold, clase):
    return list(map(lambda pair: int(pair[clase] > threshold), clf.predict_proba(data)))

big_list = []
for line in lines:
    big_list.append(dictize_tweet(line))

vocab_list = []
for dicto in big_list:
    for key in dicto.keys():
        if key not in vocab_list:
            vocab_list.append(key)

def add(a, b):
    return a + b

def add_vectors(a, b):
    return list(map(add, a, b))

def multiply_vector(v, n):
    return list(map(lambda thing: thing * n, v))

def svectorize_tweet(tweet):
    tweet_dict = dictize_tweet(tweet)
    sum_svector = [0] * 200
    uwc = 0
    for key in tweet_dict:
        key_svector = get_svector(key)
        if key_svector:
            sum_svector = add_vectors(sum_svector, multiply_vector(key_svector, tweet_dict[key]))
            uwc += tweet_dict[key]
    if uwc:
        return multiply_vector(sum_svector, 1 / uwc)
    #print(("WEEEOOOO" * 10 + "\n") * 10)
    #print(tweet)
    #print("--------")
    return sum_svector


#for testing
def train_and_eval(samples, targets, test_range_start, test_range_end):
    global acc_total
    global p_total
    global r_total
    test_samples = samples[test_range_start:test_range_end]
    train_samples = samples[:test_range_start] + samples[test_range_end:]
    if True:
        data = list(map(svectorize_tweet, test_samples + train_samples))
    else:
        data = v.fit_transform(map(dictize_tweet, test_samples + train_samples))
    test_data = data[:test_range_end - test_range_start]
    train_data = data[test_range_end - test_range_start:]
    test_targets = targets[test_range_start:test_range_end]
    train_targets = targets[:test_range_start] + targets[test_range_end:]
    clf = LogisticRegression().fit(train_data, train_targets)
    print("desired:")
    print(test_targets)
    for i in range(test_range_end - test_range_start):
        if int(test_targets[i]):
            print(test_samples[i])

    print("results:")
    predictions = new_predict(clf, test_data, 0.13, 1)
    print(predictions)
    for i in range(test_range_end - test_range_start):
        if int(predictions[i]):
            print(test_samples[i])
    cm = get_cm(test_targets, predictions)
    print(cm)
    accuracy = (cm["TP"] + cm["TN"]) / (cm["TP"] + cm["TN"] + cm["FP"] + cm["FN"])
    if cm["TP"] + cm["FP"]:
        precision = cm["TP"] / (cm["TP"] + cm["FP"])
    else:
        print(("\N{large red square}" * 10 + "\n") * 3)
        precision = 0
    recall = cm["TP"] / (cm["TP"] + cm["FN"])
    print("accuracy: " + str(accuracy))
    acc_total += accuracy
    p_total += precision
    r_total += recall
    if cm["TP"] + cm["FP"]:
        print("precision: " + str(precision))
    else:
        print("precision: NaN")
    print("recall: " + str(recall))
    print("\n\n\n")

    pairs = clf.predict_proba(test_data)
    numbers = list(map(lambda thing: thing[1], pairs))
    tweets = test_samples[:]

    numbers, tweets, pairs = zip(*sorted(zip(numbers, tweets, pairs), reverse = True))
    print("MOST TOXIC TWEETS")
    for i in range(5):
        print(tweets[i], end = "")
        print(list(pairs[i]))
        print()
    print("\n\n\n\n\n")


#cross-validation test of predict

for i in range(5):
    print("ITERATION " + str(i + 1))
    train_and_eval(lines, desired, i * 50, (i + 1) * 50)

print("AVERAGE ACCURACY:")
print(acc_total / 5)
p_avg = p_total / 5
r_avg = r_total / 5
print("AVERAGE PRECISION:")
print(p_avg)
print("AVERAGE RECALL:")
print(r_avg)
print("AVERAGE F-MEASURE:")
print(2 * p_avg * r_avg / (p_avg + r_avg))

print("\n\n\n\n\n")

# def get_new_file(words):
#     new_file = ""
#     with open ("glove.twitter.27B.200d.txt", "r") as file:
#         for line in file:
#             pair = line.split(" ", 1)
#             if pair[0] in words:
#                 new_file += line
#     return new_file
#
# the_new_file = get_new_file(vocab_list)
# with open("svectors200.txt", "w") as file:
#     file.write(the_new_file)

# #getting words by toxicity
#
# la_data = v.fit_transform(big_list)
# clf = LogisticRegression().fit(la_data, desired)
# nambari = list(clf.coef_[0])
# majina = list(v.get_feature_names_out())
#
# nambari, majina = zip(*sorted(zip(nambari, majina), reverse = True))
#
# print("MOST TOXIC WORDS")
# print(nambari[:60])
# print(majina[:60])
# print("\n\n\n\n\n")
#
# #defining toxicity finder
# diccionario = dict(zip(majina, nambari))
#
# def sort_by_toxicity(words):
#     toxicities = list(map(lambda thing: diccionario[thing], words))
#     toxicities, words = zip(*sorted(zip(toxicities, words), reverse = True))
#     return words
#
# #getting tweets by toxicity
# la_data = v.fit_transform(big_list)
# clf = LogisticRegression().fit(la_data[:75], desired[:75])
# pairs = clf.predict_proba(la_data[75:])
# numbers = list(map(lambda thing: thing[1], pairs))
# tweets = lines[75:]
#
# numbers, tweets, pairs = zip(*sorted(zip(numbers, tweets, pairs), reverse = True))
#
# print("MOST TOXIC TWEETS")
# for i in range(5):
#     print(tweets[i], end = "")
#     #print(sort_by_toxicity(list(dictize_tweet(tweets[i]).keys())))
#     print(pairs
#     [i])

#     print()

# print("DOING STUFF")
# la_data = v.fit_transform(big_list)
# clf = LogisticRegression().fit(la_data, desired)
# tweets = lines[:]
# cm = get_cm(desired, clf.predict(la_data))
# print(cm)
# print("accuracy: " + str((cm["TP"] + cm["TN"]) / (cm["TP"] + cm["TN"] + cm["FP"] + cm["FN"])))


# def get_thingy(a, b):
#     return (a - b) * (a - b)
#
# def vector_dist(a, b):
#     return sum(map(get_thingy, a, b)) ** 0.5

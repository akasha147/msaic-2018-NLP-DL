import numpy as np
import tensorflow as tf

import re

#Initialize Global variables
query_vectors = []
passage_vectors = []
labels = []



GloveEmbeddings = {}
weights_query = {}
weights_passage = {}
bias_query = {}
bias_passage = {}
weights_combined = {}
biased_combined = {}

max_query_words=12
max_passage_words=50
emb_dim=50
n_classes = 2 
batch_size = 100
n_epoches = 10
dataset_size = 500
total_size = 60000


x1 = tf.placeholder('float',[100,600])
x2 = tf.placeholder('float',[100,2500])
y  = tf.placeholder('float',[100,2])

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

weights_query = {'W_conv1': tf.Variable(tf.random_normal([3,10,1,4])),'W_conv2': tf.Variable(tf.random_normal([4,4,4,20])),'W_fc' :tf.Variable(tf.random_normal([780,300])),'out':tf.Variable(tf.random_normal([300,4]))}
weights_passage = {'W_conv1': tf.Variable(tf.random_normal([3,10,1,4])),'W_conv2': tf.Variable(tf.random_normal([4,4,4,20])),'W_fc' :tf.Variable(tf.random_normal([3380,500])),'out':tf.Variable(tf.random_normal([500,4]))}
	
bias_query = {'b_conv1':tf.Variable(tf.random_normal([4])),'b_conv2':tf.Variable(tf.random_normal([20])),'b_fc':tf.Variable(tf.random_normal([300])),'out':tf.Variable(tf.random_normal([4]))}
bias_passage = {'b_conv1':tf.Variable(tf.random_normal([4])),'b_conv2':tf.Variable(tf.random_normal([20])),'b_fc':tf.Variable(tf.random_normal([500])),'out':tf.Variable(tf.random_normal([4]))}
	
weights_combined = {'out':tf.Variable(tf.random_normal([4,2]))}
biased_combined = {'out':tf.Variable(tf.random_normal([2]))}






def loadData(fileHandle):

	for i in range(dataset_size):
		datapoint = fileHandle.readline()
		TextDataToCTF(datapoint);

def loadEmbeddings(embeddingfile):
    global GloveEmbeddings,emb_dim

    fe = open(embeddingfile,"r")
    for line in fe:
        tokens= line.strip().split()
        word = tokens[0]
        vec = tokens[1:]
        vec = " ".join(vec)
        GloveEmbeddings[word]=vec
    #Add Zerovec, this will be useful to pad zeros, it is better to experiment with padding any non-zero constant values also.
    GloveEmbeddings["zerovec"] = "0.0 "*emb_dim
    fe.close()

	

def TextDataToCTF(inputData,isEvaluation=False):
   
        tokens = inputData.strip().lower().split("\t")
        query_id,query,passage,label = tokens[0],tokens[1],tokens[2],tokens[3]

        #****Query Processing****
        words = re.split('\W+', query)
        words = [x for x in words if x] # to remove empty words 
        word_count = len(words)
        remaining = max_query_words - word_count  
        if(remaining>0):
            words += ["zerovec"]*remaining # Pad zero vecs if the word count is less than max_query_words
        words = words[:max_query_words] # trim extra words
        #create Query Feature vector 
        query_feature_vector = ""
        for word in words:
            if(word in GloveEmbeddings):
                query_feature_vector += GloveEmbeddings[word]+" "
            else:
                query_feature_vector += GloveEmbeddings['zerovec']+" " #Add zerovec for OOV terms
    
        #***** Passage Processing **********
        words = re.split('\W+', passage)
        words = [x for x in words if x] # to remove empty words 
        word_count = len(words)
        remaining = max_passage_words - word_count  
        if(remaining>0):
            words += ["zerovec"]*remaining # Pad zero vecs if the word count is less than max_passage_words
        words = words[:max_passage_words] # trim extra words
        #create Passage Feature vector 
        passage_feature_vector = ""
        for word in words:
            if(word in GloveEmbeddings):
                passage_feature_vector += GloveEmbeddings[word]+" "
            else:
                passage_feature_vector += GloveEmbeddings['zerovec']+" "
        #convert label
        label_str = 0 if label=="0" else 1 
        if isEvaluation == False :
        	labels.append(label_str)

        query_vectors.append([float(v) for v in query_feature_vector.split()])
        passage_vectors.append([float(v) for v in passage_feature_vector.split()])
        #print("|qfeatures "+query_feature_vector+" |pfeatures "+passage_feature_vector+" |labels "+label_str+"\n")


def cnn_model(x_query,x_passage):
	
	x_query= tf.reshape(x1, shape = [-1,12,50,1])
	query_conv1 = maxpool2D(convolution2d(tf.cast(x_query,tf.float32),weights_query['W_conv1'])+bias_query['b_conv1'])
	query_conv2 = maxpool2D(convolution2d(query_conv1,weights_query['W_conv2'])+bias_query['b_conv2'])
	query_fc = tf.reshape(query_conv2,[-1,780])
	query_fc = tf.nn.relu(tf.matmul(query_fc,weights_query['W_fc']+bias_query['b_fc']))
	query_fc = tf.nn.dropout(query_fc, keep_rate)
	output_query = tf.matmul(query_fc,weights_query['out'])+bias_query['out']

	x_passage = tf.reshape(x2, shape = [-1,50,50,1])
	passage_conv1 = maxpool2D(convolution2d(tf.cast(x_passage,tf.float32),weights_passage['W_conv1'])+bias_passage['b_conv1'])
	passage_conv2 = maxpool2D(convolution2d(passage_conv1,weights_passage['W_conv2'])+bias_passage['b_conv2'])
	passage_fc = tf.reshape(passage_conv2,[-1,3380])
	passage_fc = tf.nn.relu(tf.matmul(passage_fc,weights_passage['W_fc']))+bias_passage['b_fc']
	passage_fc = tf.nn.dropout(passage_fc, keep_rate)
	output_passage = tf.matmul(passage_fc,weights_passage['out'])+bias_passage['out']

	new_x = output_passage + output_query
	output = tf.matmul(new_x,weights_combined['out'])+biased_combined['out']
	
	return output


def convolution2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def maxpool2D(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")


def train_cnn_network(x1,x2):
	
	prediction = cnn_model(x1,x2)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

	with tf.Session() as sess:
		
		sess.run(tf.global_variables_initializer())
		for epoch in range(n_epoches):
			epoch_loss = 0
			for i in range(int(dataset_size/batch_size)):
				_, c = sess.run([optimizer,cost], feed_dict={x1 : query_vectors[batch_size*i:batch_size*(i+1),:],x2 : passage_vectors[batch_size*(i):batch_size*(i+1),:],y : labels[batch_size*(i):batch_size*(i+1),:] })
       			epoch_loss += c
       			print('Epoch', epoch+1, 'completed out of',n_epoches,'loss:',epoch_loss)


       			
       

       		   





if __name__ == "__main__":

    trainSetFileName = "summarized_dataset.tsv"
    validationSetFileName = "ValidationData.ctf"
    testSetFileName = "EvaluationData.ctf"
    submissionFileName = "answer.tsv"
    embeddingFileName = "glove.6B.50d.txt"

    loadEmbeddings(embeddingFileName)
    trainData=open(trainSetFileName,'r')
    
    # initWeightandBases()
    loadData(trainData)
    query_vectors = np.array(query_vectors)
    passage_vectors = np.array(passage_vectors)
    labels = np.array(([[v,1-v] for v in labels]))
    train_cnn_network(x1,x2)
    
    # print(labels)
   
    # LoadData(trainData)
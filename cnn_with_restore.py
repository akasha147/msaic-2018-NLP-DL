import numpy as np
import tensorflow as tf
import re

#Initialize Global variables
#Dataset lists
query_vectors = []
passage_vectors = []
labels = []

#dictionaries
GloveEmbeddings = {}  # to hold the glove embeddings(word-vector pair)
weights_query = {}    # weights for cnn(query_part)
weights_passage = {}  #	weights for cnn(passage_part)
bias_query = {}		  # bias for cnn(query_part)
bias_passage = {}	  #	bias for passage(passage_part)
weights_combined = {} #	weights for cnn(query_part)
biased_combined = {}  # bias for cnn(passage_part)

#constants
max_query_words=12
max_passage_words=50
emb_dim=50
n_classes = 2 
batch_size = 100
n_epoches = 10
dataset_size = 500
total_size = 60000
starting = True

#cnn constants
#for conv1 filter
filter1_dimX = 3
filter1_dimY = 10
filter1_n = 1
filter1_outn = 4

#for conv2 filter
filter2_dimX = 4
filter2_dimY = 4
filter2_n = 4
filter2_outn = 20

#for fc and out layer(query)
fc_x_query = 780
fc_y_query = 300
fc_out_query = 4

#for fc and out layer(passage)
fc_x_passage = 3380
fc_y_passage = 500
fc_out_passage = 4

keep_rate = 0.8

#for combined out layer
fc_x_combined = 4
fc_y_combined = n_classes


"""
CNN model
1)query_vector->conv_layer1->maxpool(relu)->conv_layer2->maxpool(relu)->fully_connected_query->output_query
2)passage_vector->conv_layer1->maxpool(relu)->conv_layer2->maxpool(relu)->fully_connected_passage->output_passage
3)output_query+output_passage->output
weights representation
conv_layer= [x dimension of the filter,y dimension of the filter,number of inputs,number of filters]
fc_layer = [don't know,number of ouput nodes]
output_layer = [number of output nodes,output size]
"""

"""
Read from the dataset file in 'dataset_size' chunks and process it 'batch_size' elements at a time with 'n_epoches' epoches per batch
"""
#tensorflow variables
x1 = tf.placeholder('float',[batch_size,max_query_words*emb_dim])#to hold query
x2 = tf.placeholder('float',[batch_size,max_passage_words*emb_dim])# to hold passage
y  = tf.placeholder('float',[batch_size,n_classes])# to hold labels
keep_prob = tf.placeholder(tf.float32)

weights_query = {'W_conv1': tf.Variable(tf.random_normal([filter1_dimX,filter1_dimY,filter1_n,filter1_outn])),'W_conv2': tf.Variable(tf.random_normal([filter2_dimX,filter2_dimY,filter2_n,filter2_outn])),'W_fc' :tf.Variable(tf.random_normal([fc_x_query,fc_y_query])),'out':tf.Variable(tf.random_normal([fc_y_query,fc_out_query]))}
weights_passage = {'W_conv1': tf.Variable(tf.random_normal([filter1_dimX,filter1_dimY,filter1_n,filter1_outn])),'W_conv2': tf.Variable(tf.random_normal([filter2_dimX,filter2_dimY,filter2_n,filter2_outn])),'W_fc' :tf.Variable(tf.random_normal([fc_x_passage,fc_y_passage])),'out':tf.Variable(tf.random_normal([fc_y_passage,fc_out_passage]))}
	
bias_query = {'b_conv1':tf.Variable(tf.random_normal([filter1_outn])),'b_conv2':tf.Variable(tf.random_normal([filter2_outn])),'b_fc':tf.Variable(tf.random_normal([fc_y_query])),'out':tf.Variable(tf.random_normal([fc_out_query]))}
bias_passage = {'b_conv1':tf.Variable(tf.random_normal([filter1_outn])),'b_conv2':tf.Variable(tf.random_normal([filter2_outn])),'b_fc':tf.Variable(tf.random_normal([fc_y_passage])),'out':tf.Variable(tf.random_normal([fc_out_passage]))}
	
weights_combined = {'out':tf.Variable(tf.random_normal([fc_x_combined,fc_y_combined]))}
biased_combined = {'out':tf.Variable(tf.random_normal([fc_y_combined]))}


#Load data from the file and store into np.arrays after converting them into word vectors
def loadData(fileHandle):

	for i in range(dataset_size):
		datapoint = fileHandle.readline()
		TextDataToCTF(datapoint);

#Load the embedding file into the dictionary
def loadEmbeddings(embeddingfile):
	global GloveEmbeddings,emb_dim

	fe = open(embeddingfile,"r", encoding="utf-8")
	for line in fe:
		tokens= line.strip().split()
		word = tokens[0]
		vec = tokens[1:]
		vec = " ".join(vec)
		GloveEmbeddings[word]=vec
	#For padding purpose(to max size)
	GloveEmbeddings["zerovec"] = "0.0 "*emb_dim
	fe.close()


#Takes a line from the dataset file  as input produces the corresponding word embeddings
#Populates three list -Query,Passage,Labels
def TextDataToCTF(inputData,isEvaluation=False):
   
		tokens = inputData.strip().lower().split("\t")
		query_id,query,passage,label = tokens[0],tokens[1],tokens[2],tokens[3]

		#****Query Processing****
		words = re.split('\W+', query)
		words = [x for x in words if x] 
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

		query_vectors.append([float(v) for v in query_feature_vector.split()])#Insert the entire word embedding of the query into the vector(size:1*(max_query_size*emb_dim))
		passage_vectors.append([float(v) for v in passage_feature_vector.split()])#Insert the entire word embedding of the passage into the vector(size:1*(max_passage_size*emb_dim))
		
#Definiton of the CNN model described above in tensorflow
def cnn_model(x_query,x_passage):
	
	x_query= tf.reshape(x1, shape = [-1,max_query_words,emb_dim,1])
	query_conv1 = maxpool2D(convolution2d(tf.cast(x_query,tf.float32),weights_query['W_conv1']+bias_query['b_conv1']))
	query_conv2 = maxpool2D(convolution2d(query_conv1,weights_query['W_conv2']+bias_query['b_conv2'])) 
	query_fc = tf.reshape(query_conv2,[-1,fc_x_query])
	query_fc = tf.nn.relu(tf.matmul(query_fc,weights_query['W_fc']+bias_query['b_fc']))
	query_fc = tf.nn.dropout(query_fc, keep_rate)
	output_query = tf.matmul(query_fc,weights_query['out'])+bias_query['out']

	x_passage = tf.reshape(x2, shape = [-1,max_passage_words,emb_dim,1])
	passage_conv1 = maxpool2D(convolution2d(tf.cast(x_passage,tf.float32),weights_passage['W_conv1']+bias_passage['b_conv1']))
	passage_conv2 = maxpool2D(convolution2d(passage_conv1,weights_passage['W_conv2']+bias_passage['b_conv2']))
	passage_fc = tf.reshape(passage_conv2,[-1,fc_x_passage])
	passage_fc = tf.nn.relu(tf.matmul(passage_fc,weights_passage['W_fc']))+bias_passage['b_fc']
	passage_fc = tf.nn.dropout(passage_fc, keep_rate)
	output_passage = tf.matmul(passage_fc,weights_passage['out'])+bias_passage['out']

	new_x = output_passage + output_query
	output = tf.matmul(new_x,weights_combined['out'])+biased_combined['out']
	
	return output

#Function to perform convolution on the given inputs,using given weights
def convolution2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

#Function to perform max pooling on the input(reduces the input size due to striding)
def maxpool2D(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

#Function to train the network
def train_cnn_network(x1,x2,start=False):
	
    prediction = cnn_model(x1,x2)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        if start == True:
            sess.run(tf.global_variables_initializer())
        else :
            restore = tf.train.import_meta_graph('model.ckpt.meta')
            restore.restore(sess, tf.train.latest_checkpoint('./'))
        for epoch in range(n_epoches):
            epoch_loss = 0
            for i in range(int(dataset_size/batch_size)):
                _, c = sess.run([optimizer,cost], feed_dict={x1 : query_vectors[batch_size*i:batch_size*(i+1),:],x2 : passage_vectors[batch_size*(i):batch_size*(i+1),:],y : labels[batch_size*(i):batch_size*(i+1),:] })
                epoch_loss+=c
                print('Epoch', epoch+1, 'completed out of',n_epoches,'loss:',epoch_loss)
            saver.save(sess, "./ckpt/model.ckpt")




if __name__ == "__main__":

	#Filenames
	trainSetFileName = "traindata.tsv"
    #trainSetFileName = "summarized_dataset.tsv"
	validationSetFileName = "ValidationData.ctf"
	testSetFileName = "EvaluationData.ctf"
	submissionFileName = "answer.tsv"
	embeddingFileName = "glove.6B.50d.txt"


	loadEmbeddings(embeddingFileName)#Load the file embeddings
	trainData=open(trainSetFileName,'r', encoding="utf-8")#Open the training file
	
	#Need some work here
	for i in range(10):
		loadData(trainData) #reads dataset_size(500) lines from the file
		print("Using trainingdata "+str(i))
		query_vectors = np.array(query_vectors) #Convert the list into 2D-array(shape :dataset_size*(max_query_size*emb_dim))
		passage_vectors = np.array(passage_vectors)#Convert the list into 2D-array(shape :dataset_size*(max_passage_size*emb_dim)
		labels = np.array(([[v,1-v] for v in labels]))#converts the list into 2D-array("1,0" if label=0 else="0,1" shape :dataset_size*n_classes
		if starting == False :
			train_cnn_network(x1,x2)#Call the training function
		else :
			train_cnn_network(x1,x2,True)
		#Reinitalize the vectors for the next round
		query_vectors = []
		passage_vectors = []
		labels = []
starting = False

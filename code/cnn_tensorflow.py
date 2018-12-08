import numpy as np
import tensorflow as tf
import re

#Initialize Global variables
#Dataset lists
query_vectors = []
passage_vectors = []
labels = []
train_queryid=0

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
batch_size = 2500
dataset_size = 5000
total_size = 10000
n_runs = 50
starting = True
n_unlabelled_queries = 10417
no_of_passages_per_query = 10
conv_strides = [1,1,1,1]
maxpool_strides = [1,2,2,1]

#cnn constants
#for conv1 filter
filter1_dimX = 5
filter1_dimY = 5
filter1_n = 1
filter1_outn = 4

#for conv2 filter
filter2_dimX = 3
filter2_dimY = 3
filter2_n = 4
filter2_outn = 2

#for fc and out layer(query)
fc_y_query = 20
fc_out_query = 4

#for fc and out layer(passage)
fc_y_passage = 200
fc_out_passage = 4


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

weights_query = {'W_conv1': tf.Variable(tf.random_normal([filter1_dimX,filter1_dimY,filter1_n,filter1_outn],name="qc1")),'W_conv2': tf.Variable(tf.random_normal([filter2_dimX,filter2_dimY,filter2_n,filter2_outn],name="qc2")),'out':tf.Variable(tf.random_normal([fc_y_query,fc_out_query],name="qout"))}
weights_passage = {'W_conv1': tf.Variable(tf.random_normal([filter1_dimX,filter1_dimY,filter1_n,filter1_outn],name="pc1")),'W_conv2': tf.Variable(tf.random_normal([filter2_dimX,filter2_dimY,filter2_n,filter2_outn],name="pc2")),'out':tf.Variable(tf.random_normal([fc_y_passage,fc_out_passage],name="pout"))}
	
bias_query = {'b_conv1':tf.Variable(tf.random_normal([filter1_outn],name="qbc1")),'b_conv2':tf.Variable(tf.random_normal([filter2_outn],name="qbc2")),'out':tf.Variable(tf.random_normal([fc_out_query],name="qbout"))}
bias_passage = {'b_conv1':tf.Variable(tf.random_normal([filter1_outn],name="pbc1")),'b_conv2':tf.Variable(tf.random_normal([filter2_outn],name="pbc2")),'out':tf.Variable(tf.random_normal([fc_out_passage],name="pbout"))}
	
weights_combined = {'out':tf.Variable(tf.random_normal([fc_x_combined,fc_y_combined],name="comboout"))}
biased_combined = {'out':tf.Variable(tf.random_normal([fc_y_combined],name="combooutb"))}

x1_train = tf.placeholder('float',[no_of_passages_per_query,max_query_words*emb_dim])
x2_train = tf.placeholder('float',[no_of_passages_per_query,max_passage_words*emb_dim])

#Load data from the file and store into np.arrays after converting them into word vectors
def loadData(fileHandle,readSize,isEvaluation=False):
    global dataset_size,query_vectors,passage_vectors,labels,count
    
    query_vectors = []
    passage_vectors = []
    labels = []
    for i in range(readSize):
		datapoint = fileHandle.readline()
		TextDataToCTF(datapoint,isEvaluation=isEvaluation)
			

#Load the embedding file into the dictionary
def loadEmbeddings(embeddingfile):
	global GloveEmbeddings,emb_dim
	
	fe = open(embeddingfile,"r")
	
	for line in fe:
		tokens= line.strip().split()
		word = tokens[0]
		vec = tokens[1:]
		vec = " ".join(vec)
		GloveEmbeddings[word]=vec
	
	#For padding purpose(to max size)
	GloveEmbeddings['zerovec'] = "0.0 "*emb_dim
	fe.close()


#Takes a line from the dataset file  as input produces the corresponding word embeddings
#Populates three list -Query,Passage,Labels
def TextDataToCTF(inputData,isEvaluation=False):
    
		global train_queryid,count
		
		tokens = inputData.strip().lower().split("\t")
		if isEvaluation == False:
			query_id,query,passage,label = tokens[0],tokens[1],tokens[2],tokens[3]
			train_queryid=query_id
		else:
			query_id,query,passage,passage_id =tokens[0],tokens[1],tokens[2],tokens[3]
			train_queryid=query_id
		

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
		if isEvaluation == False :
			label_str = int(label)
			labels.append(label_str)
		else:
			label_str = -1

		query_vectors.append([float(v) for v in query_feature_vector.split()])#Insert the entire word embedding of the query into the vector(size:1*(max_query_size*emb_dim))
		passage_vectors.append([float(v) for v in passage_feature_vector.split()])#Insert the entire word embedding of the passage into the vector(size:1*(max_passage_size*emb_dim))

		

#Definiton of the CNN model described above in tensorflow
def cnn_model(x_query,x_passage):
	
	x_query = tf.reshape(x_query, shape = [-1,max_query_words,emb_dim,1])
	
	query_conv1 = convolution2d(tf.cast(x_query,tf.float32),weights_query['W_conv1'])+bias_query['b_conv1']
	query_conv1 = tf.nn.leaky_relu(query_conv1)
	query_maxpool1 = maxpool2D(query_conv1)
	
	query_conv2 = convolution2d(query_maxpool1,weights_query['W_conv2'])+bias_query['b_conv2']
	query_conv2 = tf.nn.leaky_relu(query_conv2)
	query_maxpool2 = maxpool2D(query_conv2)
	
	query_fc = tf.reshape(query_maxpool2,[-1,fc_y_query])
	
	output_query = tf.nn.leaky_relu(tf.matmul(query_fc,weights_query['out'])+bias_query['out'])

	x_passage = tf.reshape(x_passage, shape = [-1,max_passage_words,emb_dim,1])
	
	passage_conv1 = convolution2d(tf.cast(x_passage,tf.float32),weights_passage['W_conv1'])+bias_passage['b_conv1']
	passage_conv1 = tf.nn.leaky_relu(passage_conv1)
	passage_maxpool1 = maxpool2D(passage_conv1)
	
	passage_conv2 = convolution2d(passage_maxpool1,weights_passage['W_conv2'])+bias_passage['b_conv2']
	passage_conv2 = tf.nn.leaky_relu(passage_conv2)
	passage_maxpool2 = maxpool2D(passage_conv2)
	
	passage_fc = tf.reshape(passage_maxpool2,[-1,fc_y_passage])
	
	output_passage = tf.nn.leaky_relu(tf.matmul(passage_fc,weights_passage['out'])+bias_passage['out'])

	new_x = output_passage * output_query
	output = tf.nn.softmax(tf.matmul(new_x,weights_combined['out'])+biased_combined['out'])
	
	return output
	


#Function to perform convolution on the given inputs,using given weights
def convolution2d(x,W):
	return tf.nn.conv2d(x,W,strides=conv_strides,padding="VALID")

#Function to perform max pooling on the input(reduces the input size due to striding)
def maxpool2D(x):
	return tf.nn.max_pool(x,ksize=maxpool_strides,strides=maxpool_strides,padding="VALID")

#Function to train the network
def train_cnn_network(trainSetFileName,saveFilePath):
	global query_vectors,passage_vectors,labels,dataset_size
	
	prediction = cnn_model(x1,x2)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=0.05).minimize(cost)
	saver = tf.train.Saver({"qc1":weights_query['W_conv1'],"qc2":weights_query['W_conv2'],"qout":weights_query['out'],"pc1":weights_passage['W_conv1'],"pc2":weights_passage['W_conv2'],"pout":weights_passage['out'],"bqc1":bias_query['b_conv1'],"bqc2":bias_query['b_conv2'],"bqout":bias_query['out'],"bpc1":bias_passage['b_conv1'],"bpc2":bias_passage['b_conv2'],"bpout":bias_passage['out'],"comboout":weights_combined['out'],"combooutb":biased_combined['out']})

	with tf.Session() as sess:
		
		sess.run(tf.global_variables_initializer())#intialize everything(like a constructor) and then restore the required variables
		print("Initialized tensorflow variables")
		
		for k in range(n_runs):
			trainData=open(trainSetFileName,'r')
			print("Run "+str(k+1)+" out of " +str(n_runs))
			
			for j in range(total_size/dataset_size):
				loadData(trainData,dataset_size) #reads dataset_size(500) lines from the file
				
				query_vectors = np.array(query_vectors) #Convert the list into 2D-array(shape :dataset_size*(max_query_size*emb_dim))
				passage_vectors = np.array(passage_vectors)#Convert the list into 2D-array(shape :dataset_size*(max_passage_size*emb_dim)
				labels = np.array(([[1-v,v] for v in labels]))
				epoch_loss = 0
					
				for i in range(int(dataset_size/batch_size)):
					_, c = sess.run([optimizer,cost], feed_dict={x1 : query_vectors[batch_size*i:batch_size*(i+1),:],x2 : passage_vectors[batch_size*(i):batch_size*(i+1),:],y : labels[batch_size*(i):batch_size*(i+1),:] })
					epoch_loss+=c
				
				print("Using trainingdata "+str(j+1)+" out of "+str(total_size/dataset_size)+" Loss:"+str(epoch_loss))
				saver.save(sess,saveFilePath)
        		

def validate_model(x1,x2,y,saveFilePath):
	
	prediction = cnn_model(x1,x2)
	correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
	saver = tf.train.Saver({"qc1":weights_query['W_conv1'],"qc2":weights_query['W_conv2'],"qout":weights_query['out'],"pc1":weights_passage['W_conv1'],"pc2":weights_passage['W_conv2'],"pout":weights_passage['out'],"bqc1":bias_query['b_conv1'],"bqc2":bias_query['b_conv2'],"bqout":bias_query['out'],"bpc1":bias_passage['b_conv1'],"bpc2":bias_passage['b_conv2'],"bpout":bias_passage['out'],"comboout":weights_combined['out'],"combooutb":biased_combined['out']})
	with tf.Session() as sess:
		saver.restore(sess,saveFilePath)
		a = accuracy.eval({x1:query_vectors,x2:passage_vectors, y:labels})
		print(a)

def predict_labels(x1_train,x2_train,testSetFileName,submissionFileName1,submissionFileName2,saveFilePath):
	global query_vectors,passage_vectors

	prediction = cnn_model(x1_train,x2_train)
	saver = tf.train.Saver({"qc1":weights_query['W_conv1'],"qc2":weights_query['W_conv2'],"qout":weights_query['out'],"pc1":weights_passage['W_conv1'],"pc2":weights_passage['W_conv2'],"pout":weights_passage['out'],"bqc1":bias_query['b_conv1'],"bqc2":bias_query['b_conv2'],"bqout":bias_query['out'],"bpc1":bias_passage['b_conv1'],"bpc2":bias_passage['b_conv2'],"bpout":bias_passage['out'],"comboout":weights_combined['out'],"combooutb":biased_combined['out']})
	testFile = open(testSetFileName,'r')
	submissionFile1 = open(submissionFileName1,'w')
	submissionFile2 = open(submissionFileName2,'w')
	queries_done = 0			
	
	for _ in range(n_unlabelled_queries):
		
		loadData(testFile,no_of_passages_per_query,isEvaluation=True)
		
		query_vectors = np.array(query_vectors)
		passage_vectors =np.array(passage_vectors)
		
		print(str(train_queryid)+" "+str(queries_done))
		if queries_done != 0:
			submissionFile1.write("\n")
			submissionFile2.write("\n")
		submissionFile1.write(train_queryid)
		submissionFile2.write(train_queryid)

		
		with tf.Session() as sess:
			saver.restore(sess, saveFilePath)
			
			pred = prediction.eval({x1_train:query_vectors,x2_train:passage_vectors})
        	for entry in pred:
        		if entry[0]>entry[1]:
        			submissionFile1.write("\t0")
        		else:
        			submissionFile1.write("\t1")
        		submissionFile2.write("\t"+str(entry[1]))

        	query_vectors = []
        	passage_vectors = []
        	queries_done+=1


if __name__ == "__main__":

	#Filenames
    trainSetFileName = "../../data/oversampled_data.tsv"
    validationSetFileName = "../../data/validationdata2.tsv"
    testSetFileName = "../../data/eval1_unlabelled.tsv"
    submissionFileName1 = "../answer/answer_binary2.tsv"
    submissionFileName2 = "../answer/answerk.tsv"
    embeddingFileName = "../../data/glove.6B.50d.txt"
    saveFilePath = "../ckpt/model_40percent.ckpt"
    
    loadEmbeddings(embeddingFileName)
    

    ##Uncomment anyone functionality at a time
    
    ##This is for validation(validates one batch of trainingdata only)
    # trainData=open(trainSetFileName,'r')
    # loadData(trainData,batch_size) 
    # query_vectors = np.array(query_vectors) 
    # passage_vectors = np.array(passage_vectors)
    # labels = np.array(([[v,1-v] for v in labels]))
    # validate_model(x1,x2,y,saveFilePath)
    
    ##This is for training
    train_cnn_network(trainSetFileName,saveFilePath)

    # ##This is for prediction
    # predict_labels(x1_train,x2_train,testSetFileName,submissionFileName1,submissionFileName2,saveFilePath)
import time
import csv
import pickle
import numpy as np
import operator
import tensorflow as tf
# Load .csv dataset
def pre_data():
    
    with open("./dataset/dataset-train-diginetica/train-item-views.csv", "r") as f:
        reader = csv.DictReader(f, delimiter=';')
        sess_clicks = {}
        sess_date = {}
        ctr = 0
        curid = -1
        curdate = None
        for data in reader:
            sessid = data['sessionId'] #session id
            if curdate and not curid == sessid:
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
                sess_date[curid] = date
            curid = sessid
            item = data['itemId'] #item id
            curdate = data['eventdate'] #time
            if  sessid in sess_clicks:#sess_clicks.has_key(sessid):
                sess_clicks[sessid] += [item]
            else:
                sess_clicks[sessid] = [item]
            ctr += 1
            if ctr % 100000 == 0:
                print ('Loaded', ctr)
        print('total loaded sessions are:', sessid)
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        sess_date[curid] = date
    
    # Filter out length 1 sessions
    store_clicks = {}
    store_date = {}
    for s in sess_clicks.keys():
        if len(sess_clicks[s]) != 1:
            store_clicks[s]=sess_clicks[s]
            store_date[s]=sess_date[s]
    sess_clicks=store_clicks
    sess_date=store_date
    
    # Count number of times each item appears
    iid_counts = {}
    for s in sess_clicks:
        seq = sess_clicks[s]
        for iid in seq:
            if iid in iid_counts:
                iid_counts[iid] += 1
            else:
                iid_counts[iid] = 1
    
    # Shows how many times each item has appeared in all sessions in a sorted manner
    sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

    # Filter out those items appear less than 5 times in all sessions
    store_clicks={}
    store_date={}
    for s in sess_clicks.keys():
        curseq = sess_clicks[s]
        #Filter out those items that shows less than 5 times in total
        filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
        # Keep those session that longer than 2 items after filtering
        if len(filseq) >= 2:
            store_clicks[s]=filseq
            store_date[s]=sess_date[s]
    # Update sess_clisks and sess_date
    sess_clicks=store_clicks
    sess_date=store_date

    # Split out test set based on dates
    dates = list(sess_date.items())
    maxdate = dates[0][1]
    mindate = dates[0][1]
    
    for _, date in dates:
        if maxdate < date:
            maxdate = date
        if mindate > date:
            mindate = date
    
    # Split out the last 1 days for testset
    splitdate = maxdate - 86400 * 1
    print('\nThe first session date is:\n', time.gmtime(mindate))
    print('\nThe last session date is:\n', time.gmtime(maxdate))
    print('\nSplit date is:\n', time.gmtime(splitdate))
    train_sess = list(filter(lambda x: x[1] < splitdate, dates))
    test_sess = list(filter(lambda x: x[1] > splitdate, dates))
    
    # Sort sessions by date
    train_sess = sorted(train_sess, key=operator.itemgetter(1))
    test_sess = sorted(test_sess, key=operator.itemgetter(1))
    
    # Convert training sessions to sequences and renumber items to start from 1
    item_dict = {}
    item_ctr = 1
    train_seqs = []
    train_dates = []

    for s, date in train_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_seqs += [outseq]
        train_dates += [date]
    print("\ntotal number of sessions in the training set is:", len(train_sess))
    
    test_seqs = []
    test_dates = []
    # Convert test sessions to sequences, ignoring items that do not appear in training set
    for s, date in test_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_seqs += [outseq]
        test_dates += [date]
    print("total number of sessions in the test set is:", len(test_sess))
    print("\ntotal number of different items in the training set is:", len(item_dict))
    
    #Data augmentation and labels setting
    def process_seqs(iseqs, idates):
        out_seqs = []
        out_dates = []
        labs = []
        for seq, date in zip(iseqs, idates):
            for i in range(1, len(seq)):
                tar = seq[-i]
                labs += [tar]
                out_seqs += [seq[:-i]]
                out_dates += [date]
    
        return out_seqs, out_dates, labs
    
    
    tr_seqs, tr_dates, tr_labs = process_seqs(train_seqs,train_dates)
    te_seqs, te_dates, te_labs = process_seqs(test_seqs,test_dates)
    
    train = (tr_seqs, tr_labs)
    test = (te_seqs, te_labs)

    print('\ntotal number of sessions in the training set after augmentation is: ', len(tr_seqs))
    print('\ntotal number of sessions in the test set after augmentation is: ', len(te_seqs))
    
    #f1 = open('train.csv', 'wb')
    #pickle.dump(train, f1)
    #f1.close()
    #f2 = open('test.csv', 'wb')
    #pickle.dump(test, f2)
    #f2.close()
    
    print('Done.')
    
    #Preprocess for input of tensorflow
    '''
    Create the matrices from the datasets.
    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.
    if maxlen is set, we will cut all sequence to this maximum
    lenght.
    This swap the axis!
    '''
    ##############training set############
    lengths = [len(i) for i in tr_seqs]
    maxlen=np.max(lengths)
    n_samples = len(tr_seqs)

    x = np.zeros((maxlen, n_samples))
    X_mask = np.ones((maxlen, n_samples))
    for idx, s in enumerate(tr_seqs):
        x[-lengths[idx]:, idx] = s
    
    X_mask *= (1 - (x == 0))
    
    X=np.transpose(x)
    X_mask=np.transpose(X_mask)

    def get_minibatches_idx(n, minibatch_size, shuffle=False):
        """
        Used to shuffle the dataset at each iteration.
        """
    
        idx_list = np.arange(n, dtype="int32")
    
        if shuffle:
            np.random.shuffle(idx_list)
    
        minibatches = []
        minibatch_start = 0
        for i in range(n // minibatch_size):
            minibatches.append(idx_list[minibatch_start:
                                        minibatch_start + minibatch_size])
            minibatch_start += minibatch_size
    
        if minibatch_start != n:
            # Make a minibatch out of what is left
            minibatches.append(idx_list[minibatch_start:])
    
        return zip(range(len(minibatches)), minibatches)
    
    aa=get_minibatches_idx(len(X), 512, shuffle=False)
    b=0
    bb=[]
    X = X.astype(int)
    
    for _, train_index in aa:
        bb.append(train_index)
        b+=1
    bb.remove(bb[-1])
    labs=np.array(tr_labs)
    print('\nNumber of minibatches is: ', len(bb))
    
    #################test set##############################
    lengths = [len(i) for i in te_seqs]
    n_samples_test = len(te_seqs)
    x_test = np.zeros((maxlen, n_samples_test))
    X_test_mask = np.ones((maxlen, n_samples_test))
    for idx, s in enumerate(te_seqs):
        x_test[-lengths[idx]:, idx] = s
    
    X_test_mask *= (1 - (x_test == 0))
    
    X_test=np.transpose(x_test)
    
    X_test_mask=np.transpose(X_test_mask)
    aa_test=get_minibatches_idx(len(X_test), 512, shuffle=False)
    b_test=0
    bb_test=[]
    X_test=X_test.astype(int)
    X_test = X_test[:,-19:] 
    X_mask= X_mask[:,-19:] 
    X_test_mask = X_test_mask[:,-19:] 
    X = X[:,-19:] 
    for _, train_index in aa_test:
        bb_test.append(train_index)
        b_test+=1
    bb_test.remove(bb_test[-1])
    labs_test=np.array(te_labs)
    return X,X_test,labs,labs_test,bb,bb_test,X_mask,X_test_mask
    
#X,X_t,Y,Y_t = pre_data()
def get_batch_data():
    # Load data
    X,X_t,Y,Y_t,_,_= pre_data()
    
    # calc total batch count
    num_batch = len(X) // 512
    
    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y])
            
    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=512, 
                                capacity=512*64,   
                                min_after_dequeue=512*32, 
                                allow_smaller_final_batch=False)
    
    return x, y, num_batch,input_queues # (N, T), (N, T), ()

#A,B,C,D = get_batch_data()


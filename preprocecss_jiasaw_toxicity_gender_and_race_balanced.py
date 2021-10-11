import csv
import random

#identity_individual_annotations.csv is downloaded from https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data.
toxicity_worker_demographics = open("identity_individual_annotations.csv","r")
reader_toxicity_worker_demographics = csv.reader(toxicity_worker_demographics)

line_num = -1

# race_list = {}
#  
# for row in reader_toxicity_worker_demographics:
#     line_num += 1
#     if line_num == 0:
#         continue
#     temp_worker_id = row[0]
#     temp_race = row[2]
#      
#     if temp_race in race_list:
#         race_list[temp_race] += 1
#     else:
#         race_list[temp_race] = 0
#          
# print(race_list)

#row[3] 'male': 209165, 'female': 233568, 'male female': 137516, 'transgender': 30473,  


workid2gender_dic = {}
for row in reader_toxicity_worker_demographics:
    line_num += 1
    if line_num == 0:
        continue
    
    temp_worker_id = row[0]
    temp_race = row[3]
    #row[4] is race
    
    if temp_race == "male" or temp_race == "female" or temp_race == "male female" or temp_race == "transgender":
    #if temp_race == "asian" or temp_race == "black" or temp_race == "white" or temp_race == "latino":
        if temp_worker_id not in workid2gender_dic:
            workid2gender_dic[temp_worker_id] = temp_race
    else:
        continue
    
print("length of workid2gender_dic:" + str(len(workid2gender_dic)))

line_num = -1

num_non_offensive_train = 0
num_offensive_train = 0
num_non_offensive_test = 0
num_offensive_test = 0

num_non_offensive_asian_train = 0
num_non_offensive_black_train = 0
num_non_offensive_white_train = 0
num_non_offensive_latino_train = 0

num_non_offensive_asian_test = 0
num_non_offensive_black_test = 0
num_non_offensive_white_test = 0
num_non_offensive_latino_test = 0



num_yes_offensive_asian_train = 0
num_yes_offensive_black_train = 0
num_yes_offensive_white_train = 0
num_yes_offensive_latino_train = 0

num_yes_offensive_asian_test = 0
num_yes_offensive_black_test = 0
num_yes_offensive_white_test = 0
num_yes_offensive_latino_test = 0

train_no_num = 0
train_yes_num = 0
valid_no_num = 0
valid_yes_num = 0
test_no_num = 0
test_yes_num = 0

num_one = 0
num_zero = 0

train_num_threshould = 10528 
test_num_threshould = 1052
class_num = 4

import torch

# Load an En-Fr Transformer model trained on WMT'14 data :
en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe')

# Use the GPU (optional):
en2ru.cuda()

num_total_words = 1024
num_words = 1024

num_1000_words = 0
threshoud_num_words = 1024

train_csvfile = open("./train_jigsaw_gender.csv","w")
writer_train = csv.writer(train_csvfile)
test_csvfile = open("./test_jigsaw_gender.csv","w")
writer_test = csv.writer(test_csvfile)

#train.csv is downloaded from https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data.
toxicity_comments_train = open("train.csv","r")
reader_toxicity_comments_train = csv.reader(toxicity_comments_train)


yes_asian_id2content_dic = {}
yes_black_id2content_dic = {}
yes_white_id2content_dic = {}
yes_latino_id2content_dic = {}
no_asian_id2content_dic = {}
no_black_id2content_dic = {}
no_white_id2content_dic = {}
no_latino_id2content_dic = {}

train_test_num = 0

for row in reader_toxicity_comments_train:
    line_num += 1
    if line_num == 0:
        continue
    
    temp_worker_id = row[0]
    temp_label = "0"
    temp_toxic = float(row[1])
    if temp_toxic > 0.5:
        temp_label = "1"
    #temp_target = row[1]
    temp_text = row[2]
    
    if temp_worker_id in workid2gender_dic:
        temp_gender = workid2gender_dic[temp_worker_id]
        
        train_test_num += 1
        
        
        if temp_label == "0":
            #if train_no_num > 12528:
            #    continue
            train_no_num += 1
            if temp_gender == "male":
                num_non_offensive_asian_train += 1
            if temp_gender == "female":
                num_non_offensive_black_train += 1
            if temp_gender == "male female":
                num_non_offensive_white_train += 1
            if temp_gender == "transgender":
                num_non_offensive_latino_train += 1

        
        if temp_label == "1":
            #if train_yes_num > 12528:
            #    continue
            train_yes_num += 1
            if temp_gender == "male":
                num_yes_offensive_asian_train += 1
            if temp_gender == "female":
                num_yes_offensive_black_train += 1
            if temp_gender == "male female":
                num_yes_offensive_white_train += 1
            if temp_gender == "transgender":
                num_yes_offensive_latino_train += 1

        #if train_yes_num > 12528 and train_no_num > 12528:
        #   break
        final_text = ""
        #doc_words = re.split(r'[;,\s]\s*', temp_text_2)
        #doc_words_old = re.split(r'[;,\s]\s*', temp_text_2)
        #print("doc_words_old:" + str(len(doc_words_old)))
        doc_words = en2ru.tokenize(temp_text)
        print("doc_words:" + str(len(doc_words)))
        #doc_words = temp_text_2.split(" ")
        if len(doc_words) > num_total_words:
            print("ha")
            print("before:" + str(len(doc_words)))
            if len(doc_words) > threshoud_num_words:
                num_1000_words += 1
            last_1024_words = doc_words[0:num_words]
            print("after:" + str(len(last_1024_words)))
            final_text = " ".join(last_1024_words)
            #print(len(final_text.split(" ")))
        else:
            final_text = temp_text
         
	#valid dataset will be generated based on the train datset created here in the MixText code. As for MixText code, we refer to the published code in github.
        if train_no_num <= train_num_threshould and temp_label == "0":
         
            #writer_train_no.writerow([int(temp_label), temp_gender, final_text])
            #train_csvfile_no.flush()
             
            if temp_gender == "male":
                #writer_train_sexist_twiter_asian_no.writerow([int(temp_label) + 1, temp_gender, final_text])
                #train_csvfile_sexist_twiter_asian_no.flush()
                yes_asian_id2content_dic[train_test_num] = [int(temp_label) + 1, temp_gender, final_text]
            if temp_gender == "female":
                #writer_train_sexist_twiter_black_no.writerow([int(temp_label) + 1, temp_gender, final_text])
                #train_csvfile_sexist_twiter_black_no.flush()
                yes_black_id2content_dic[train_test_num] = [int(temp_label) + 1, temp_gender, final_text]
            if temp_gender == "male female":
                #writer_train_sexist_twiter_white_no.writerow([int(temp_label) + 1, temp_gender, final_text])
                #train_csvfile_sexist_twiter_white_no.flush()
                yes_white_id2content_dic[train_test_num] = [int(temp_label) + 1, temp_gender, final_text]
            if temp_gender == "transgender":
                #writer_train_sexist_twiter_white_no.writerow([int(temp_label) + 1, temp_gender, final_text])
                #train_csvfile_sexist_twiter_white_no.flush()
                yes_latino_id2content_dic[train_test_num] = [int(temp_label) + 1, temp_gender, final_text]
             
        if train_yes_num <= train_num_threshould and temp_label == "1":
         
            #writer_train_yes.writerow([int(temp_label), temp_gender, final_text])
            #train_csvfile_yes.flush()               
                 
            if temp_gender == "male":
                #writer_train_sexist_twiter_asian_yes.writerow([int(temp_label) + 1, temp_gender, final_text])
                #train_csvfile_sexist_twiter_asian_yes.flush()
                no_asian_id2content_dic[train_test_num] = [int(temp_label) + 1, temp_gender, final_text]
            if temp_gender == "female":
                #writer_train_sexist_twiter_black_yes.writerow([int(temp_label) + 1, temp_gender, final_text])
                #train_csvfile_sexist_twiter_black_yes.flush()
                no_black_id2content_dic[train_test_num] = [int(temp_label) + 1, temp_gender, final_text]
            if temp_gender == "male female":
                #writer_train_sexist_twiter_white_yes.writerow([int(temp_label) + 1, temp_gender, final_text])
                #train_csvfile_sexist_twiter_white_yes.flush()
                no_white_id2content_dic[train_test_num] = [int(temp_label) + 1, temp_gender, final_text]
            if temp_gender == "transgender":
                #writer_train_sexist_twiter_white_yes.writerow([int(temp_label) + 1, temp_gender, final_text])
                #train_csvfile_sexist_twiter_white_yes.flush()
                no_latino_id2content_dic[train_test_num] = [int(temp_label) + 1, temp_gender, final_text]
         
         
         
 
 
        if not (train_no_num <= train_num_threshould and temp_label == "0") and not (train_yes_num <= train_num_threshould and temp_label == "1"):# and train_test_num <= train_num_threshould + test_num_threshould:
            if temp_label == "0":
                test_no_num += 1
                if temp_gender == "male":
                    num_non_offensive_asian_test += 1
                if temp_gender == "female":
                    num_non_offensive_black_test += 1
                if temp_gender == "male female":
                    num_non_offensive_white_test += 1
                if temp_gender == "transgender":
                    num_non_offensive_latino_test += 1
                     
                final_text = ""
                #doc_words = re.split(r'[;,\s]\s*', temp_text_2)
                #doc_words_old = re.split(r'[;,\s]\s*', temp_text_2)
                #print("doc_words_old:" + str(len(doc_words_old)))
                doc_words = en2ru.tokenize(temp_text)
                print("doc_words:" + str(len(doc_words)))
                #doc_words = temp_text_2.split(" ")
                if len(doc_words) > num_total_words:
                    print("ha")
                    print("before:" + str(len(doc_words)))
                    if len(doc_words) > threshoud_num_words:
                        num_1000_words += 1
                    last_1024_words = doc_words[0:num_words]
                    print("after:" + str(len(last_1024_words)))
                    final_text = " ".join(last_1024_words)
                    #print(len(final_text.split(" ")))
                else:
                    final_text = temp_text
 
                if test_no_num <= test_num_threshould:
                      
                    writer_test.writerow([int(temp_label) + 1, temp_gender, final_text])
                    test_csvfile.flush()
 
            if temp_label == "1":
                test_yes_num += 1
                if temp_gender == "male":
                    num_yes_offensive_asian_test += 1
                if temp_gender == "female":
                    num_yes_offensive_black_test += 1
                if temp_gender == "male female":
                    num_yes_offensive_white_test += 1  
                if temp_gender == "transgender":
                    num_yes_offensive_latino_test += 1      
          
                final_text = ""
                #doc_words = re.split(r'[;,\s]\s*', temp_text_2)
                #doc_words_old = re.split(r'[;,\s]\s*', temp_text_2)
                #print("doc_words_old:" + str(len(doc_words_old)))
                doc_words = en2ru.tokenize(temp_text)
                print("doc_words:" + str(len(doc_words)))
                #doc_words = temp_text_2.split(" ")
                if len(doc_words) > num_total_words:
                    print("ha")
                    print("before:" + str(len(doc_words)))
                    if len(doc_words) > threshoud_num_words:
                        num_1000_words += 1
                    last_1024_words = doc_words[0:num_words]
                    print("after:" + str(len(last_1024_words)))
                    final_text = " ".join(last_1024_words)
                    #print(len(final_text.split(" ")))
                else:
                    final_text = temp_text
                  
                if test_yes_num <= test_num_threshould:
                      
                    writer_test.writerow([int(temp_label) + 1, temp_gender, final_text])
                    test_csvfile.flush()
             
 
 
temp_int_ha = int(train_num_threshould / class_num)
print("temp_int_ha: " + str(temp_int_ha))
 
#selecet_or_not = False
 
for i in range(temp_int_ha):
    #if i >= 25:
    selecet_or_not = False
    while selecet_or_not == False:
        temp_int = random.randint(1,train_num_threshould * 2)
        if temp_int in yes_asian_id2content_dic:
            temp_need2write = yes_asian_id2content_dic[temp_int]
            writer_train.writerow(temp_need2write)
            selecet_or_not = True
   
             
    selecet_or_not = False
    while selecet_or_not == False:
        temp_int = random.randint(1,train_num_threshould * 2)
        if temp_int in yes_black_id2content_dic:
            temp_need2write = yes_black_id2content_dic[temp_int]
            writer_train.writerow(temp_need2write)
            selecet_or_not = True
               
    selecet_or_not = False
    while selecet_or_not == False:
        temp_int = random.randint(1,train_num_threshould * 2)
        if temp_int in yes_white_id2content_dic:
            temp_need2write = yes_white_id2content_dic[temp_int]
            writer_train.writerow(temp_need2write)
            selecet_or_not = True
              
    selecet_or_not = False
    while selecet_or_not == False:
        temp_int = random.randint(1,train_num_threshould * 2)
        if temp_int in yes_latino_id2content_dic:
            temp_need2write = yes_latino_id2content_dic[temp_int]
            writer_train.writerow(temp_need2write)
            selecet_or_not = True
               
               
    selecet_or_not = False
    while selecet_or_not == False:
        temp_int = random.randint(1,train_num_threshould * 2)
        if temp_int in no_asian_id2content_dic:
            temp_need2write = no_asian_id2content_dic[temp_int]
            writer_train.writerow(temp_need2write)
            selecet_or_not = True
               
    selecet_or_not = False
    while selecet_or_not == False:
        temp_int = random.randint(1,train_num_threshould * 2)
        if temp_int in no_black_id2content_dic:
            temp_need2write = no_black_id2content_dic[temp_int]
            writer_train.writerow(temp_need2write)
            selecet_or_not = True
               
    selecet_or_not = False
    while selecet_or_not == False:
        temp_int = random.randint(1,train_num_threshould * 2)
        if temp_int in no_white_id2content_dic:
            temp_need2write = no_white_id2content_dic[temp_int]
            writer_train.writerow(temp_need2write)
            selecet_or_not = True
  
    selecet_or_not = False
    while selecet_or_not == False:
        temp_int = random.randint(1,train_num_threshould * 2)
        if temp_int in no_latino_id2content_dic:
            temp_need2write = no_latino_id2content_dic[temp_int]
            writer_train.writerow(temp_need2write)
            selecet_or_not = True


print("num_non_offensive_asian_train " + str(num_non_offensive_asian_train))
print("num_non_offensive_black_train " + str(num_non_offensive_black_train))
print("num_non_offensive_white_train " + str(num_non_offensive_white_train))
print("num_non_offensive_latino_train " + str(num_non_offensive_latino_train))

print("num_yes_offensive_asian_train " + str(num_yes_offensive_asian_train))
print("num_yes_offensive_black_train " + str(num_yes_offensive_black_train))
print("num_yes_offensive_white_train " + str(num_yes_offensive_white_train))
print("num_yes_offensive_latino_train " + str(num_yes_offensive_latino_train))

print("num_non_offensive_asian_test " + str(num_non_offensive_asian_test))
print("num_non_offensive_black_test " + str(num_non_offensive_black_test))
print("num_non_offensive_white_test " + str(num_non_offensive_white_test))
print("num_non_offensive_latino_test " + str(num_non_offensive_latino_test))

print("num_yes_offensive_asian_test " + str(num_yes_offensive_asian_test))
print("num_yes_offensive_black_test " + str(num_yes_offensive_black_test))
print("num_yes_offensive_white_test " + str(num_yes_offensive_white_test))
print("num_yes_offensive_latino_test " + str(num_yes_offensive_latino_test))


    
print("num_1000_words: " + str(num_1000_words))    
print("Done!")   






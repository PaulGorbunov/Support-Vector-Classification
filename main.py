'''
Support Vector Classification
'''
import matplotlib.pyplot as plt
import tensorflow as tf
from multiprocessing import Process,Pipe
import pickle
import os
import time
import shutil
from sklearn.svm import SVC
import pandas as pd

class classify:
    
    def __init__(self):
        try:
            shutil.rmtree('data')
            os.makedirs('data')
            (self.x_train,self.y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
            self.length = len(self.y_train)
            self.beg = 60000
            self.fas = 10000
        except :
            pass 
        
    
    def main(self):
        start_time = time.time()
        #self.processor(self.write)
        self.write(0,self.length)
        del self.x_train
        del self.y_train
        self.processor(self.get)
        self.bust('satur')
        self.bust('inv')
        
        print("--- %s seconds ---" % (time.time() - start_time))
        self.combile_files()
    
    def combile_files(self):
        sat_o = []  # Create an empty dictionary
        sat_f = []
        inv_o = []
        inv_f = []
        info_o = []
        info_f = []
        l1 = int(self.length/self.fas)
        for i in range(l1):
            lab = str(self.fas*i)
            with open('data/satur_one'+lab, 'rb') as f:
                d1 = pickle.load(f)
                sat_o = sat_o + d1
            with open('data/satur_five'+lab, 'rb') as f1:
                d2 = pickle.load(f1)
                sat_f = sat_f + d2
            with open('data/inv_one'+lab, 'rb') as f2:
                d1 = pickle.load(f2)
                inv_o = inv_o + d1
            with open('data/inv_five'+lab, 'rb') as f3:
                d2 = pickle.load(f3)
                inv_f = inv_f + d2
            with open('data/one'+lab, 'rb') as f:
                info_o += pickle.load(f)
            with open('data/five'+lab, 'rb') as f:
                info_f += pickle.load(f)
                
            os.remove('data/satur_one'+lab)
            os.remove('data/satur_five'+lab)
            os.remove('data/inv_one'+lab)
            os.remove('data/inv_five'+lab)
            os.remove('data/one'+lab)
            os.remove('data/five'+lab)
            
        with open('data/one','wb') as f:
            pickle.dump(info_o,f)
        del info_o
        with open('data/five','wb') as f:
            pickle.dump(info_f,f)
        del info_f
        
        feat_one = list(zip(sat_o,inv_o))
        feat_five = list(zip(sat_f,inv_f))
        y_one = [1 for i in range(len(feat_one))]
        y_five = [5 for i in range(len(feat_five))]
        del sat_o,sat_f,inv_f,inv_o
        svclassifier = SVC(kernel='linear')  
        svclassifier.fit(feat_one+feat_five,y_one+y_five) 
        with open('svc','wb') as f:
            pickle.dump(svclassifier,f)
        
    def bust(self,lable):
        ar_len = self.length
        if lable == 'satur':
            function = self.satur 
        else:
            function = self.inverse
        pro = []
        ite = int(ar_len/self.fas)
        for i in range (ite):
            pro.append(Process(target=self.analise, args=(function,self.fas*i,lable)))
        for pr in pro:
            pr.start()
        for y in pro:
            y.join()
            
    
    def analise(self,funct,b,lable):
        p_1,c_1 = Pipe()
        p_2,c_2 = Pipe()
        with open('data/one'+str(b),"rb") as f:
            o_lis = pickle.load(f)
        with open('data/five'+str(b),"rb") as f:
            f_lis = pickle.load(f)
        one_p = Process(target=funct, args=(o_lis,'one',c_1))
        five_p = Process(target=funct, args=(f_lis,'five',c_2))
        one_p.start()
        five_p.start()
        one_p.join()
        five_p.join()
        ans = [p_1.recv(),p_2.recv()]
        for q in ans:
            lab = q[0]
            with open('data/'+lable+'_'+lab+str(b),'wb') as fi:
                pickle.dump(q[1:],fi)
    
    def inverse(self,lis,lab,conn):
        iv = []
        for u in lis:
            su = int(0)
            l1 = len(u)
            for t in range(len(u)):
                for y in range(l1):
                    if (u[y][t] == u[y][-1*(t+1)]):
                        su += 1
            iv.append(su)
        
        conn.send([lab]+iv)
        conn.close()
        
    def satur(self,lis,lab,conn):
        intence = []
        for u in lis:
            s = 0
            for y in u:
                s = s + sum(y)
            intence.append(s)
        
        conn.send([lab]+intence)
        conn.close()
    
    def processor(self,funct):
        ar_len = self.length
        pro = []
        for i in range (int(ar_len/self.beg)):
            pro.append(Process(target=funct, args=(self.beg*i,self.beg*i+self.beg)))
        for pr in pro:
            pr.start()
        for y in pro:
            y.join()
        
    def get(self,b,e):
        with open('data/lable'+str(b)+"_"+str(e),"rb") as f:
            lis = pickle.load(f)
        with open('data/img'+str(b)+"_"+str(e),"rb") as f:
            l_img = pickle.load(f)
            
        l1 = int(self.beg/self.fas)
        proc = []
        for t in range(l1):
            proc.append(Process(target=self.filednum ,args=(b+self.fas*t , lis[self.fas*t:self.fas*t+self.fas], l_img[self.fas*t:self.fas*t+self.fas])))
        for pro in proc:
            pro.start()
        for y in proc:
            y.join()
            
        os.remove('data/img'+str(b)+"_"+str(e))
        os.remove('data/lable'+str(b)+"_"+str(e))
        
    def filednum(self,lab,lis,l_img):
        one_five = [['one'],['five']]
        k = len(lis)
        for i in range(k):
            if (lis[i] == 1):
                one_five[0].append(i)
            if (lis[i] == 5):
                one_five[1].append(i)
        
        for i in one_five :
            p = []
            for ind in i[1:]:
                p.append(l_img[ind])
            with open('data/'+i[0]+str(lab),'wb') as e:
                pickle.dump(p,e)
        
    def write(self,b,e):
        with open('data/img'+str(b)+"_"+str(e),'wb') as fl:  
            pickle.dump(self.x_train[b:e],fl)
        with open('data/lable'+str(b)+"_"+str(e),'wb')as fe:
            pickle.dump(self.y_train[b:e],fe);
    
    def show(self,num):
        plt.imshow(num, cmap='Greys')
        plt.show()
                
class test:
    def __init__(self):
        (x_train,y_train),(self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
    
    def main(self):
        self.lable = []
        for i in range(len(self.y_test)):
            if self.y_test[i] == 1 or self.y_test[i] == 5:
                self.lable.append(i)
        
        self.t_l = []
        for y in self.lable:
            self.t_l.append(self.x_test[y])

        
    def features (self,matr):
        su = 0
        l1 = len(matr)
        for t in range(len(matr)):
            for y in range(l1):
                if (matr[y][t] == matr[y][-1*(t+1)]):
                    su += 1
                    s = 0
        s = 0
        for y in matr:
            s = s + sum(y)
        
        return [s,su]
    
    def predictor(self):
        s = 0
        with open('svc','rb') as f:
            svc = pickle.load(f)
            
        for i in range(len(self.t_l)):
            fea = self.features(self.t_l[i])
            p = svc.predict([fea])[0]
            ans = self.y_test[self.lable[i]]
            if p == ans :
                s += 1
        print (s,'out of',len(self.lable))
        print ("% is",(s*100)/len(self.lable))
            
    
    def say(self,num):
        with open('svc','rb') as f:
            svc = pickle.load(f)
        fe = self.features(self.t_l[num])
        print (svc.predict([fe])[0])
        self.show(self.t_l[num])
        
        
        
    def show(self,num):
        plt.imshow(num, cmap='Greys')
        plt.show()
    
if __name__ == "__main__":
    cl = classify()
    cl.main()
    #best case --- 5.81206750869751 seconds ---
    #worst case --- 6.792078971862793 seconds ---
            
        
    

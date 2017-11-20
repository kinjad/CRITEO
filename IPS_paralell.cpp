#include<iostream>
#include<map>
#include<cstdlib>
#include<cmath>
#include<sstream>
#include<vector>
#include<string>
#include<fstream>
#include <thread>

#define NUM_THREADS 10

using namespace std;
double feat_freq[74000], pos_freq[74000], neg_freq[74000], feat_max[74000];

void readData(string file, vector<vector<vector<int> > > &data_idx, vector<vector<vector<double> > > &data_val, vector<double> &data_reward, vector<double> &data_ips){
// ips: inverse propensity score, reward = 1 if clicked, data_val[i][j][k]: i-th impression, j-th ad, (data_idx[i][j][k])-th feature value

	ifstream f_in(file.c_str());
	string buffer="";

	getline(f_in,buffer);	
	while(buffer.size()>0){
		istringstream s_in(buffer);
		double reward, ips, val;
		int idx;
		vector<int> feat_idx;
		vector<double> feat_val;

		vector<vector<int> > scen_idx;
		vector<vector<double> > scen_val;

		string tmp;

		s_in >> tmp >> reward >> tmp >> ips >> tmp;

		map<int,int> is_pos;
		is_pos.clear();
		while(s_in >> idx >> val){
			feat_idx.push_back(idx);
			feat_val.push_back(val);
			feat_freq[idx]++;

			if(feat_max[idx] < val)
				feat_max[idx] = val;
			if(reward < 0.5)
				is_pos[idx] = 1;
		}
		scen_idx.push_back(feat_idx);
		scen_val.push_back(feat_val);

		vector<int> neg_idx;
		while(getline(f_in,buffer) && buffer[0]=='f'){
			feat_idx.clear();
			feat_val.clear();
			istringstream sin(buffer);
			sin >> tmp;
			while(sin >> idx >> val){
				feat_idx.push_back(idx);
				feat_val.push_back(val);

				feat_freq[idx]++;
				
				is_pos[idx] = 0;

				if(feat_max[idx] <val) 
					feat_max[idx] = val;
			}
		
			scen_idx.push_back(feat_idx);
			scen_val.push_back(feat_val);
		}

		if(reward < 0.5 && data_ips.size() < 107546)
		for(int k = 0; k < scen_idx[0].size(); k++)
			if(is_pos[scen_idx[0][k]]>0) 
				pos_freq[scen_idx[0][k]]++;

		data_idx.push_back(scen_idx);
		data_val.push_back(scen_val);
		
		if(reward > 0.5) reward = 0.0;
		else reward = 1;	

		data_reward.push_back(reward);
		data_ips.push_back(ips);
	}
	cout << "Read file done" << endl;
}

void computeGradient(vector<double> w, double &x, vector<double> &g, vector<vector<vector<int> > > data_idx, vector<vector<vector<double> > > data_val, vector<double> data_reward, vector<double> data_ips, int start, int end, int &hN){
	int N = data_reward.size();
	x = 0;
	//int hN = 0;
	for(int i = 0; i < g.size(); i++)
		g[i] = 0;

	for(int i = start; i < end; i++){
		int M = data_idx[i].size();
		vector<double> f_values;

		f_values.clear();
		for(int j = 0; j < M; j++){
			double f = 0;
			for(int k = 0; k < data_idx[i][j].size(); k++)
				f += w[data_idx[i][j][k]] * (1.0 + (data_val[i][j][k] > 1)); // / feat_max[data_idx[i][j][k]];
			f_values.push_back(f);
		}

		//For the overflow issue
		double max_value = -1e10;
		for(int j = 0; j < M; j++)
			if(max_value < f_values[j])
				max_value = f_values[j];

		double Z = 0;
		for(int j = 0; j < M; j++){
			f_values[j] -= max_value;
			f_values[j] = exp(f_values[j]);
			Z += f_values[j];
		}

		double p = f_values[0]/Z;	

		if(data_reward[i] > 0.5){
			double _x =  p * data_ips[i] * data_reward[i];
			x += _x;
			hN++;

			for(int j = 0; j < M; j++){
                        	for(int k = 0; k < data_idx[i][j].size(); k++)
                                	g[data_idx[i][j][k]] -= _x * f_values[j] / Z * (1.0 + (data_val[i][j][k] > 1)); // / feat_max[data_idx[i][j][k]];
                	}
			
			for(int k = 0; k < data_idx[i][0].size(); k++)
				g[data_idx[i][0][k]] += _x * (1.0 + (data_val[i][0][k] > 1)); // / feat_max[data_idx[i][0][k]];					
		
			
		}else{
			double _x = 10.0 * p * data_ips[i] * data_reward[i];
			x += _x;
			hN += 10;

			//reward shaping for gradient calculation
			_x = 10.0 * p * data_ips[i] * (data_reward[i]-0.002); 

			for(int j = 0; j < M; j++){
                        	for(int k = 0; k < data_idx[i][j].size(); k++)
                                	g[data_idx[i][j][k]] -= _x * f_values[j] / Z * (1.0 + (data_val[i][j][k] > 1)); // / feat_max[data_idx[i][j][k]];
                	}
			
			for(int k = 0; k < data_idx[i][0].size(); k++)
				g[data_idx[i][0][k]] += _x * (1.0 + (data_val[i][0][k] > 1)); // / feat_max[data_idx[i][0][k]];					
	
		}

	}
	//x /= (double)hN;

	//for(int k = 0; k < g.size(); k++)
	//	g[k] /= (double)hN;
	//cout << "IN" << hN << endl;
}

int main(int argc, char** argv){


        string filename = argv[1];
	vector<vector<vector<int> > > data_idx;
	vector<vector<vector<double> > > data_val; 
	vector<double> data_reward;
	vector<double> data_ips;	
	vector<double> w;

	readData(filename, data_idx, data_val, data_reward, data_ips);  

	thread threads[NUM_THREADS], test_thread;


	int mut_thr = 30000, pos_thr = 50, neg_thr = 600;
	for(int i = 0; i < 74000; i++){
//		if(pos_freq[i] < pos_thr && neg_freq[i] < neg_thr)
		//if(feat_freq[i] < mut_thr || pos_freq[i] < pos_thr)
//			w.push_back(0);
//		else 
			w.push_back(0.001);
//		cout << pos_freq[i] << " " << neg_freq[i] << endl;
	}	

	int total_num = data_ips.size();
	int num_iter = 500;
	double lr = 0.05, reg = 0.01;
	int share_each_thread = total_num /3*2 / NUM_THREADS;
	for(int t = 0; t < num_iter; t++){
	        double x_test;
		double x_train[NUM_THREADS + 1] = {0}, x_train_total = 0;
		int hN[NUM_THREADS + 1] = {0}, hN_test = 0, hN_train = 0;
		vector<double> g = w;
		vector<vector<double> > gs(NUM_THREADS, vector<double>(74000, 0));		


		test_thread = thread(computeGradient, w, ref(x_test), ref(g), ref(data_idx), ref(data_val), ref(data_reward), ref(data_ips), total_num/3*2, total_num, ref(hN_test));

	    
		//		computeGradient(w, x_test, g, data_idx, data_val, data_reward, data_ips, total_num/3*2, total_num, hN_test);
		
		//x_test /= (double)hN_test;
		

		for (int i = 0; i < NUM_THREADS; i++) {
		int start = i * share_each_thread, end = (i + 1) * share_each_thread;		
		threads[i] = thread(computeGradient, w, ref(x_train[i]), ref(gs[i]), ref(data_idx), ref(data_val), ref(data_reward), ref(data_ips), start, end, ref(hN[i]));
		}
		

		for (int i = 0; i < NUM_THREADS; i++) {
		  threads[i].join();    
		}
		//test_thread.join();

		for (int i = 0; i < NUM_THREADS; i++) {
		  x_train_total += x_train[i];
		  hN_train += hN[i];
		  //cout << hN[i] << " " << x_train[i] << endl;
		}

		for (int j = 0; j < gs[0].size(); j++) {
		  g[j] = 0;
		  for (int i = 0; i < NUM_THREADS; i++) {
		    g[j] += gs[i][j];
		  }
		}

		for (int k = 0; k < g.size(); k++)
		  g[k] /= (double)hN_train;
		x_train_total /= (double)hN_train;

		test_thread.join();
		x_test /= (double)hN_test;

		/*
		computeGradient(w, x_train_total, g, data_idx, data_val, data_reward, data_ips, 0, total_num/3*2, hN_train);

		for (int k = 0; k < g.size(); k++)
		  g[k] /= (double)hN_train;

		x_train_total /= hN_train;
		*/

		
		for(int i = 0; i < w.size(); i++)
		     //if(feat_freq[i] >= mut_thr && pos_freq[i] >= pos_thr)
		     //if( (pos_freq[i] >= pos_thr && neg_freq[i] < neg_thr)){ // || (pos_freq[i] < pos_thr && neg_freq[i] >= neg_thr))
		     if( (pos_freq[i] >= pos_thr)){ // || (pos_freq[i] < pos_thr && neg_freq[i] >= neg_thr))
			//w[i] += lr*10000*g[i] - reg*(0.01*((w[i]>0) - (w[i]<0)));
			//w[i] += lr*10000*g[i] - reg*(w[i] + 0.1*((w[i]>0) - (w[i]<0)));
			w[i] += lr*10000*g[i] - reg*w[i];
			if(w[i] < 0) w[i] = 0;
//		     }else if ((pos_freq[i] < pos_thr && neg_freq[i] >= neg_thr))
//		     {
//			w[i] += lr*10000*g[i] - reg*w[i];
//			if(w[i] > 0) w[i] = 0;
		     }
		     else w[i] = 0;
		cout << "Iter " << t << ": Train Obj value = " << x_train_total*10000 << " Test Obj value = " << x_test*10000 << endl;
	}
		

	return 0;
}




  


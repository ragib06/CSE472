/*
Title : Naive Bayes implementation
Date  : 07.12.2011
Auther: Ragib Ahsan, Dhiman Paul, Riyad Parvez
Course: CSE472, Assignment-3
Note  : Cross Validation not done properly :(
		Instead of partitioning test data, test data is partition randomly !
*/

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cmath>
#include <ctime>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <iterator>
using namespace std;

#define MAXLINESIZE 200
#define MAXDOCUMENTS 7000
#define MAXWORDS 50000
#define MAXSTORYSIZE 10000
#define MAXCLASS 200
#define MAXTEST 2500
#define RANDMAX 50
#define run 50

#define PI 2*acos(0.)
#define EPS 1e-7
#define MAX(a,b) (a)>(b)?(a):(b)
#define MIN(a,b) (a)<(b)?(a):(b)

#define IN  freopen("training.data","r",stdin);
#define IN1  freopen("test.data","r",stdin);
#define OUT  freopen("out.txt","w",stdout);

#define DELIM ",./ 0123456789\n(){}[]\""

#define HAM		0
#define EUCLID	1
#define TFIDF	2

#define TRAIN	true
#define TEST	false

#define SMOOTH_FACT 1.
#define IDFTHRESH 2.5

struct wordinfo{
    int first;      //word id
    int second;     //frequency
    double third;   //tfidf weight
    wordinfo(){third=second=first=0;}
    wordinfo(int wid, int freq){first = wid; second = freq; third = 0.;}
};

struct document{
	double tinorm;			//TF-IDF vector norm(magnitude)
	string topic;			//document topic
	vector<wordinfo > words;

	document(){tinorm=0;}
	document(const char *str){topic = string(str);tinorm=0;}

    int find(int key){
        int l=0, r=words.size()-1,m;
        while(l<=r){
            m = (l+r)/2;
            if(words[m].first > key){
                r = m-1;
            }else if(words[m].first < key){
                l = m+1;
            }else{
                return m;
            }
        }
        return -1;
    }
};


map<string,int > dict;				//global dictionary
vector<document > docs,testdocs;	//training and test documents
vector<int> randtest;
double IDF[MAXWORDS];
vector<pair<int, double> > hdm;		//distance measure
vector<pair<int, double> > edm;		//distance measure
vector<pair<int, double> > tdm;		//distance measure

map<string, int> toplist;           //topic list
vector<int> topclass[MAXCLASS];      //classes with their documents
int TCT[MAXCLASS];
int STk[MAXWORDS][MAXCLASS];


//comparison function for sorting pairs<docid, dist>
bool dcomp(pair<int, double> a, pair<int, double> b){
	return a.second < b.second;
}

bool fcomp(wordinfo a, wordinfo b){
	return a.first < b.first;
}

int gfreq[MAXWORDS];	//global frequency list of words
int ndocs;				//number of training documents
int tdocs;				//number of test documents
int ntwords;			//number of total words
int ntopics;            //number of different topics


//returns a document object starting from "line" input
//determines training/testing by boolean argument "train"
document load_document(bool train, char *line){
	string top(line);			//topic
	getchar();					//blank line

	gets(line);					//title

	char c = getchar();			//blank line
	if(c != '\n') gets(line);	//if the line wasn't blank gobble the line [wrong input]

	gets(line);					//location, date
	getchar();					//blank line

	char s[MAXSTORYSIZE]={0};	//store the story text here
	while(gets(line) != NULL){	//read story text
		if(strlen(line) == 0)	break;	//end of story text
		strcat(s,line);
		strcat(s,"\n");
	}

	if(strlen(s) == 0) {return document("");}	//return empty document on empty story text

	document d;

	d.topic = top;

	char *p = strtok(s,DELIM);
	while(p != NULL){					//parse the story text for words
		string t(p);					//a word
		transform(t.begin(), t.end(), t.begin(), ::tolower);	//convert to lower case

		int w,dt = dict[t];
		if(dt == 0){					//new word in dictionary
			dt = ntwords;
			dict[t] = dt;				//add the word to the dictionary

			d.words.push_back(wordinfo(ntwords,1));	//add the word to the documents word-list
			gfreq[ntwords]++;			//new word
			if(train)   ntwords++;

		}else{							//existing word in dictionary
		    for(w=0;w < d.words.size() && d.words[w].first != dt;w++);

			if(w == d.words.size()){
				d.words.push_back(wordinfo(dt, 1));	//add the word to word-list
				if(train)	gfreq[dt]++;	//?????
			}else{						//old word
                d.words[w].second++;
			}
		}
		p = strtok(NULL,DELIM);
	}
	return d;
}

//loads training/testing data
void load_data(bool train)
{
	if(train){					//clear dictionary, global frequency list
		memset(gfreq,0,sizeof(gfreq));
		memset(IDF,0,sizeof(IDF));
		memset(TCT,0,sizeof(TCT));
		dict.clear();
		toplist.clear();
	}

	char line[MAXLINESIZE];
	int i,j;

	if(train){
		ndocs = 0;
		ntwords = 1;
		ntopics = 1;
	}else{
        tdocs = 0;
	}

	while(gets(line) != NULL){				//read input line by line
		if(strlen(line) == 0)	continue;	//skip blank line
		document d = load_document(train, line);	//load a document [train/test]
		if(d.topic.size() == 0 || d.words.size() == 0)	continue;	//skip the document if empty

        //insert into topic-list and topic-class
        if(train){
            if(toplist.find(d.topic) != toplist.end()){
                topclass[toplist[d.topic]].push_back(docs.size());
            }else{
                toplist[d.topic] = ntopics;
                topclass[ntopics].push_back(docs.size());
                ntopics++;
            }
        }

		if(train){							//add document to specified list
			docs.push_back(d);
			ndocs++;
		}else{
			testdocs.push_back(d);
			tdocs++;
		}
	}

	if(train)	for(i=0;i<ndocs;i++)	sort(docs[i].words.begin(), docs[i].words.end(), fcomp);		//sort the words in the documents
	else	    for(i=0;i<tdocs;i++)	sort(testdocs[i].words.begin(), testdocs[i].words.end(), fcomp);


	if(train){		//calculate the TF-IDF weights for the words in the documents
		for(i=0;i<ndocs;i++){
			for(j=0;j<(signed)docs[i].words.size();j++){
				double TF = docs[i].words[j].second / (docs[i].words.size() * 1.);
                int w = docs[i].words[j].first;
                if(!IDF[w]) IDF[ w ] = log((1.*ndocs)/gfreq[ w ]);
				docs[i].words[j].third = TF*IDF[w];
				docs[i].tinorm += TF*TF*IDF[w]*IDF[w];
			}
			docs[i].tinorm = sqrt(docs[i].tinorm);
		}
		return;
	}

	for(i=0;i<tdocs;i++){
		for(j=0;j<(signed)testdocs[i].words.size();j++){
			double TF = testdocs[i].words[j].second / (testdocs[i].words.size() * 1.);
			int w = testdocs[i].words[j].first;
			if(!IDF[w])  IDF[w] = log((1.*ndocs)/gfreq[ testdocs[i].words[j].first ]);
			testdocs[i].words[j].third = TF*IDF[w];
			testdocs[i].tinorm += TF*TF*IDF[w]*IDF[w];
		}
		testdocs[i].tinorm = sqrt(testdocs[i].tinorm);
	}

}

//test the algo for distance measure provided by 'testid' with 'ntd' test datas
double test(int ntd){
	int i,j,k,s,ind,cnt[3][10]={0};
	register int l;
	double hdist,edist,tdist,d,a,b,acc[3][10];
	vector<wordinfo >::iterator pit;
	vector<wordinfo > v(1000+2);

	for(i=0;i<ntd;i++){		//for all testdata
		//hdm.clear();				//clear the distance measure vector
		//edm.clear();
		tdm.clear();
		document *testdoc = &testdocs[ randtest[i] ];	//current test data

		for(j=0;j<ndocs;j++){	//for all training data
			tdist = edist = hdist = 0.;
			document *traindoc = &docs[j];	//current training data

            pit = merge(traindoc->words.begin(),traindoc->words.end(),testdoc->words.begin(),testdoc->words.end(),v.begin(),fcomp);
            s = int(pit-v.begin());

            for(l=0;l<s;l++){
                //int a=0,b=0;

                //a = v[l].second;
                if(  (l < s-1) && (v[l].first == v[l+1].first) ){
                    //b=v[l+1].second;
                    tdist += v[l].third * v[l+1].third;
                    l++;
                }
                //else   hdist++;

                //d = a-b;
                //edist += (d*d);
            }
            //edist = sqrt(edist);

            tdist /= traindoc->tinorm;
            tdist /= testdoc->tinorm;    //now dist = cos(theta)
            tdist = acos(tdist);          //now dist = theta


			//hdm.push_back(make_pair(j,hdist));	//training documents with corresponding distance from current testdata
            //edm.push_back(make_pair(j,edist));
            tdm.push_back(make_pair(j,tdist));
		}

		//sort(hdm.begin(), hdm.end(), dcomp);		//sort thetraining documents in ascending order of distance
        //sort(edm.begin(), edm.end(), dcomp);
        sort(tdm.begin(), tdm.end(), dcomp);

        map<string, int> fr[3];        //record topic frequencies of nearest neighbours
		for(k=5;k<=5;k+=1){          //choosing values for k
		    int max[3] = {0};
		    int p[3] = {0};
			for(l=0;l<k;l++){       //for k nearest neighbours in dm
                //string htop = docs[hdm[l].first].topic;   //topic of the l'th nearest neighbour
			    //string etop = docs[edm[l].first].topic;
			    string ttop = docs[tdm[l].first].topic;

			    //fr[HAM][ htop ]++ ;           //count topic frequency
			    //fr[EUCLID][ etop ]++ ;
			    fr[TFIDF][ ttop ]++ ;
/*
			    if(fr[HAM][ htop ] > max[HAM]){    //record the max
                    max[HAM] = fr[HAM][ htop ];
                    p[HAM] = hdm[l].first;
			    }

			    if(fr[EUCLID][ etop ] > max[EUCLID]){    //record the max
                    max[EUCLID] = fr[EUCLID][ etop ];
                    p[EUCLID] = edm[l].first;
			    }
*/
			    if(fr[TFIDF][ ttop ] > max[TFIDF]){    //record the max
                    max[TFIDF] = fr[TFIDF][ ttop ];
                    p[TFIDF] = tdm[l].first;
			    }
			}
/*
			if(docs[p[HAM]].topic == testdoc->topic){		//matches topic!
                cnt[HAM][k]++;
            }
            if(docs[p[EUCLID]].topic == testdoc->topic){		//matches topic!
                cnt[EUCLID][k]++;
            }
*/            if(docs[p[TFIDF]].topic == testdoc->topic){		//matches topic!
                cnt[TFIDF][k]++;
            }
		}
	}

/*    char method[40];

    strcpy(method,"Hamming Distance");
    printf("\n\n%-18s:\n",method);
    printf("------------------\n");
    for(k=1;k<=7;k+=1){
        acc[HAM][k] = (cnt[HAM][k]*100.)/ntd;
        printf("Accuracy (k=%d)    : %.2lf%%\n",k,acc[HAM][k]);
    }

    strcpy(method,"Euclidean Distance");
    printf("\n\n%-18s:\n",method);
    printf("------------------\n");
    for(k=1;k<=7;k+=1){
        acc[EUCLID][k] = (cnt[EUCLID][k]*100.)/ntd;
        printf("Accuracy (k=%d)    : %.2lf%%\n",k,acc[EUCLID][k]);
    }

    strcpy(method,"Cosine Similarity");
    printf("\n\n%-18s:\n",method);
    printf("------------------\n");
*/
    for(k=5;k<=5;k+=1){
        acc[TFIDF][k] = (cnt[TFIDF][k]*100.)/ntd;
        //printf("Accuracy (k=%d)    : %.2lf%%\n",k,acc[TFIDF][k]);
    }
    return acc[TFIDF][5];
}

void calcBayesData(){
    int i,j,k,l;

    for(i=1;i<ntopics;i++){
        int  tct = 0;
        for(j=0;j<topclass[i].size();j++){
            document *d = &docs[ topclass[i][j] ];
            for(k=0;k < d->words.size();k++){
                if(IDF[ d->words[k].first ] < IDFTHRESH  || IDF[ d->words[k].first ] > 8.5) continue;
                tct += d->words[k].second;
            }
        }
        TCT[i] = tct + ntwords;
    }

    int sum,sz;
    for(i=1;i<dict.size();i++){
        for(j=0;j<ntopics;j++){
            sum = 0;
            sz = topclass[j].size();
            for(k=0;k<sz;k++){
                document *d = &docs[ topclass[j][k] ];
                int ind = d->find( i );
                if(ind != -1) sum += d->words[ind].second;
            }
            STk[i][j] = sum;
        }
    }
}

double naiveBayes(){
    int t,tp,w,d,ind,cnt,c,succ=0;
    double Pr,SPtk,NV,Pmax;

    for(t=0;t<randtest.size();t++){
        document td = testdocs[ randtest[t] ];
        Pmax = -9999999.;
        for(tp=1;tp<ntopics;tp++){
            SPtk=0;
            for(w=0;w<td.words.size();w++){
                if(IDF[ td.words[w].first ] < IDFTHRESH || IDF[ td.words[w].first ] > 8.5)   continue;
                cnt = STk[ td.words[w].first ][tp];
                SPtk += log((cnt+SMOOTH_FACT)/TCT[tp]);

            }
            Pr = topclass[tp].size();
            Pr /= ndocs;

            NV = log(Pr) + SPtk;

            if(NV > Pmax){
                Pmax = NV;
                c = tp;
            }
        }
        //printf("%d %d %.2lf\n",toplist[td.topic], c, Pmax);
        if(toplist[td.topic] == c){
            succ++;
        }
    }

    //printf("\n\n%-18s:\n","Naive Bayes");
    //printf("------------------\n");
    //printf("Accuracy          : %.2lf%%\n",(succ*100.)/randtest.size());
    return (succ*100.)/randtest.size();
}

int main(int argc, char **argv)
{
	OUT
	
    srand(time(NULL));
	time_t start,end;
	int i;

	IN  //redirect standard input to training data file
	load_data(TRAIN);
	printf("Training docs     : %d\n",ndocs);
    printf("Train words       : %d\n",ntwords);

	IN1 //redirect standard input to testing data file
	load_data(TEST);
	//printf("test data loaded!\n");
	printf("Test docs         : %d\n",tdocs);
	printf("Total words       : %d\n",dict.size());
	fflush(stdout);

	//printf("the: %.2lf\n",IDF[ dict["go"] ]);


    calcBayesData();
    printf("Total classes     : %d\n\n",ntopics);



    time (&start);

    //int ntd = tdocs;

    double acc[2][run+2], tst[run+2];

    tst[0] = acc[0][0] = acc[1][0] = 0.;
    //int run = 5;
    for(i=1;i<=run;i++){

        randtest.clear();
        while(true){
            int n = rand()%tdocs;
            randtest.push_back(n);
            if(randtest.size() == RANDMAX)  break;
        }

        acc[0][i] = test(randtest.size());
        acc[1][i] = naiveBayes();

        acc[0][0] += acc[0][i];
        acc[1][0] += acc[1][i];

        tst[i] = acc[1][i] - acc[0][i];
        tst[0] += tst[i];
    }
    acc[0][0] = acc[0][0]/run;
    acc[1][0] = acc[1][0]/run;
    tst[0] = tst[0]/run;

    printf("mean(cosine)      : %.2lf%%\n", acc[0][0]);
    printf("mean(bayes)       : %.2lf%%\n\n", acc[1][0]);

    double sd[2]={0};
    double tsd=0.;
    for(i=1;i<=run;i++){
        sd[0] += (acc[0][i]-acc[0][0])*(acc[0][i]-acc[0][0]);
        sd[1] += (acc[1][i]-acc[1][0])*(acc[1][i]-acc[1][0]);
        tsd += (tst[i] - tst[0])*(tst[i] - tst[0]);
    }
    sd[0] = sqrt(sd[0]/run);
    sd[1] = sqrt(sd[1]/run);
    tsd = sqrt(tsd/(run*(run-1)));

    printf("sd(cosine)        : %.2lf%%\n", sd[0]);
    printf("sd(bayes)         : %.2lf%%\n\n", sd[1]);

    printf("t-test(0.005)     : %.2lf%% %.2lf%%\n", tst[0]-2.680*tsd, tst[0]+2.680*tsd);
    printf("t-test(0.01)      : %.2lf%% %.2lf%%\n", tst[0]-2.405*tsd, tst[0]+2.405*tsd);
    printf("t-test(0.05)      : %.2lf%% %.2lf%%\n", tst[0]-1.677*tsd, tst[0]+1.677*tsd);

    printf("\n");
	time (&end);
	int dif = difftime (end,start);
	printf("Execution time    : ");
	if(dif>=60) printf("%dm %ds\n",dif/60,dif%60);
    else        printf("%ds\n",dif);


	return 0;
}

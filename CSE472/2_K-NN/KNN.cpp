/*
Title : K-NN Algorithm implementation
Date  : 26.10.2011
Auther: Ragib Ahsan, Dhiman Paul, Riyad Parvez
Course: CSE472, Assignment-2
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

#define PI 2*acos(0.)
#define EPS 1e-7
#define MAX(a,b) (a)>(b)?(a):(b)
#define MIN(a,b) (a)<(b)?(a):(b)

#define IN  freopen("training.data","r",stdin);
#define IN1  freopen("test.data","r",stdin);
#define OUT  freopen("out.txt","w",stdout);

#define DELIM ",./ 0123456789\n\""

#define HAM		0
#define EUCLID	1
#define TFIDF	2

#define TRAIN	true
#define TEST	false

bool flag = true;

struct wordinfo{
    int first;
    int second;
    double third;
    wordinfo(){third=second=first=0;}
    wordinfo(int wid, int freq){first = wid; second = freq; third = 0.;}
};

struct document{
	double tinorm;			//TF-IDF vector norm(magnitude)
	string topic;			//document topic
	vector<wordinfo > words;

	document(){tinorm=0;}
	document(const char *str){topic = string(str);tinorm=0;}
};


map<string,int > dict;				//global dictionary
vector<document > docs,testdocs;	//training and test documents
vector<pair<int, double> > hdm;		//distance measure
vector<pair<int, double> > edm;		//distance measure
vector<pair<int, double> > tdm;		//distance measure
//double IDF[MAXWORDS];

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
int nwords;				//number of total words


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
	//d.freqs.clear();

	char *p = strtok(s,DELIM);
	while(p != NULL){					//parse the story text for words
		string t(p);					//a word
		transform(t.begin(), t.end(), t.begin(), ::tolower);	//convert to lower case

		int w,dt = dict[t];
		if(dt == 0){					//new word in dictionary
			dt = nwords;
			dict[t] = dt;				//add the word to the dictionary

			d.words.push_back(wordinfo(nwords,1));	//add the word to the documents word-list
			//d.freqs[nwords] = 1;		//first encounter of the word
			gfreq[nwords]++;			//new word
			nwords++;

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
		dict.clear();
	}

	char line[MAXLINESIZE];
	int i,j;

	if(train){
		ndocs = 0;
		nwords = 1;
	}else 	tdocs = 0;

	while(gets(line) != NULL){				//read input line by line
		if(strlen(line) == 0)	continue;	//skip blank line
		document d = load_document(train, line);	//load a document [train/test]
		if(d.topic.size() == 0)	continue;	//skip the document if empty
		if(train){							//add document to specified list
			docs.push_back(d);
			ndocs++;
		}else{
			testdocs.push_back(d);
			tdocs++;
		}
	}

	if(train)	for(i=0;i<ndocs;i++)	sort(docs[i].words.begin(),docs[i].words.end(), fcomp);		//sort the words in the documents
	else	for(i=0;i<tdocs;i++)	sort(testdocs[i].words.begin(),testdocs[i].words.end(), fcomp);

//printf("%.2lf\n",docs[0].words[0].third);

	if(train){		//calculate the TF-IDF weights for the words in the documents
		for(i=0;i<ndocs;i++){
			for(j=0;j<(signed)docs[i].words.size();j++){
				double TF = docs[i].words[j].second / (docs[i].words.size() * 1.);
				double IDF = log((1.*ndocs)/gfreq[ docs[i].words[j].first ]);
				//docs[i].tiw[ docs[i].words[j].first ] = TF*IDF;
				docs[i].words[j].third = TF*IDF;
				docs[i].tinorm += TF*TF*IDF*IDF;
				gfreq[nwords];
			}
			docs[i].tinorm = sqrt(docs[i].tinorm);
		}
		return;
	}

	for(i=0;i<tdocs;i++){
		for(j=0;j<(signed)testdocs[i].words.size();j++){
			double TF = testdocs[i].words[j].second / (testdocs[i].words.size() * 1.);
			double idf = log((1.*ndocs)/gfreq[ testdocs[i].words[j].first ]);
			//testdocs[i].tiw[ testdocs[i].words[j].first ] = TF*idf;
			testdocs[i].words[j].third = TF*idf;
			testdocs[i].tinorm += TF*TF*idf*idf;
		}
		testdocs[i].tinorm = sqrt(testdocs[i].tinorm);
	}

}

//test the algo for distance measure provided by 'testid' with 'ntd' test datas
void test(int ntd){
	//printf("testing %d test docs...\n",ntd);
	int i,j,k,s,ind,cnt[3][10]={0};
	register int l;
	double hdist,edist,tdist,d,a,b,acc[3][10];
	vector<wordinfo >::iterator pit;

	for(i=0;i<ntd;i++){		//for all testdata
		hdm.clear();				//clear the distance measure vector
		edm.clear();
		tdm.clear();
		document *testdoc = &testdocs[i];	//current test data

		for(j=0;j<ndocs;j++){	//for all training data
			tdist = edist = hdist = 0.;
			document *traindoc = &docs[j];	//current training data

            vector<wordinfo > v(testdoc->words.size()+traindoc->words.size()+2);

            pit = merge(traindoc->words.begin(),traindoc->words.end(),testdoc->words.begin(),testdoc->words.end(),v.begin(),fcomp);
            s = int(pit-v.begin());

            for(l=0;l<s;l++){
                int a=0,b=0;

                a = v[l].second;
                if(  (l < s-1) && (v[l].first == v[l+1].first) ){
                    b=v[l+1].second;
                    tdist += v[l].third * v[l+1].third;
                    l++;
                }else   hdist++;

                d = a-b;
                edist += (d*d);
            }
            edist = sqrt(edist);

            tdist /= traindoc->tinorm;
            tdist /= testdoc->tinorm;    //now dist = cos(theta)
            tdist = acos(tdist);          //now dist = theta


			hdm.push_back(make_pair(j,hdist));	//training documents with corresponding distance from current testdata
            edm.push_back(make_pair(j,edist));
            tdm.push_back(make_pair(j,tdist));
		}

		sort(hdm.begin(), hdm.end(), dcomp);		//sort thetraining documents in ascending order of distance
        sort(edm.begin(), edm.end(), dcomp);
        sort(tdm.begin(), tdm.end(), dcomp);

        map<string, int> fr[3];        //record topic frequencies of nearest neighbours
		for(k=1;k<=7;k+=1){          //choosing values for k
		    int max[3] = {0};
		    int p[3] = {0};
			for(l=0;l<k;l++){       //for k nearest neighbours in dm
                string htop = docs[hdm[l].first].topic;   //topic of the l'th nearest neighbour
			    string etop = docs[edm[l].first].topic;
			    string ttop = docs[tdm[l].first].topic;

			    fr[HAM][ htop ]++ ;           //count topic frequency
			    fr[EUCLID][ etop ]++ ;
			    fr[TFIDF][ ttop ]++ ;

			    if(fr[HAM][ htop ] > max[HAM]){    //record the max
                    max[HAM] = fr[HAM][ htop ];
                    p[HAM] = hdm[l].first;
			    }

			    if(fr[EUCLID][ etop ] > max[EUCLID]){    //record the max
                    max[EUCLID] = fr[EUCLID][ etop ];
                    p[EUCLID] = edm[l].first;
			    }

			    if(fr[TFIDF][ ttop ] > max[TFIDF]){    //record the max
                    max[TFIDF] = fr[TFIDF][ ttop ];
                    p[TFIDF] = tdm[l].first;
			    }
			}

			if(docs[p[HAM]].topic == testdoc->topic){		//matches topic!
                cnt[HAM][k]++;
            }
            if(docs[p[EUCLID]].topic == testdoc->topic){		//matches topic!
                cnt[EUCLID][k]++;
            }
            if(docs[p[TFIDF]].topic == testdoc->topic){		//matches topic!
                cnt[TFIDF][k]++;
            }
		}
	}

    char method[40];

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
    for(k=1;k<=7;k+=1){
        acc[TFIDF][k] = (cnt[TFIDF][k]*100.)/ntd;
        printf("Accuracy (k=%d)    : %.2lf%%\n",k,acc[TFIDF][k]);
    }

}

int main(int argc, char **argv)
{
	time_t start,end;

	OUT

	IN  //redirect standard input to training data file
	load_data(TRAIN);
	printf("Training docs     : %d\n",ndocs);
	printf("Total words       : %d\n",dict.size());

	IN1 //redirect standard input to testing data file
	load_data(TEST);
	//printf("test data loaded!\n");
	printf("Test docs         : %d\n",tdocs);
	//printf("Total words       : %d\n",dict.size());
	fflush(stdout);

	
	time (&start);

    int ntd = tdocs;
	if(argc == 2)   ntd = atoi(argv[1]);
	test(ntd);

	time (&end);
	int dif = difftime (end,start);
	printf("\nExecution time    : ");
	if(dif>=60) printf("%dm %ds\n",dif/60,dif%60);
    else        printf("%ds\n",dif);

	return 0;
}

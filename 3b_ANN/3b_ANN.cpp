/*
Title : 2-Layer Neural Network
Date  : 21.12.2011
Auther: Ragib Ahsan, Dhiman Paul, Riyad Parvez
Course: CSE472, Assignment-3b
*/

#include <iostream>
#include <cmath>
#include <cstring>
#include <string>
#include <algorithm>
#include <stack>
#include <queue>
#include <vector>
#include <cctype>
#include <fstream>
#include <numeric>
#include <map>
#include <iterator>
#include <cstdlib>
#include <cstdio>
#include <ctime>
using namespace std;

#define INF 99999999
#define EPS 1e-7
#define MIN(a,b) ((a)<(b))?(a):(b)
#define MAX(a,b) ((a)>(b))?(a):(b)
#define REP(i,n) for(i=0; i<(n); i++)
#define FOR(i,a,b) for(i=(a); i<=(b); i++)
#define SET(t,v) memset((t), (v), sizeof(t))
#define sz size()
#define pb push_back
#define i64 long long
#define ALL(x) x.begin(), x.end()


#define SIZE 100+10

#define IN1 freopen("DataGD\\1.LearningBooleanFunctions\\XORFunc.dat","r",stdin);
#define IN2 freopen("DataGD\\1.LearningBooleanFunctions\\AnotherFunc.dat","r",stdin);
#define IN3 freopen("DataGD\\2.LearningRealValuedFunctions\\Data.dat","r",stdin);
#define IN4 freopen("DataGD\\3.LearningClassificaitonTask\\data.csv","r",stdin);

#define OUT1 freopen("DataGD\\1.LearningBooleanFunctions\\XORFunc_out.txt","w",stdout);
#define OUT2 freopen("DataGD\\1.LearningBooleanFunctions\\AnotherFunc_out.txt","w",stdout);
#define OUT3 freopen("DataGD\\2.LearningRealValuedFunctions\\Data_out.txt","w",stdout);
#define OUT4 freopen("DataGD\\3.LearningClassificaitonTask\\data_out.txt","w",stdout);

#define BOOL 0
#define REAL 1
#define CLASS 2

#define DELIM " ,"
#define MM 5
#define MVALS 3,5,10,20,50

#define MAX_IN_NODES    10
#define MAX_HIDDEN_NODES    55


vector< vector<int> >       BoolData;   //data for boolean values stored here
vector< vector<double> >    RealData;   //data for real values stored here
vector< vector<double> >    ClassData;  //data for class values stored here

double _L1W[MAX_IN_NODES][MAX_HIDDEN_NODES];
double _L2W[MAX_HIDDEN_NODES];

double L1W[MAX_IN_NODES][MAX_HIDDEN_NODES];
double L2W[MAX_HIDDEN_NODES];

double oK,delK;
double oH[MAX_HIDDEN_NODES], delH[MAX_HIDDEN_NODES];

int MV[MM] = {MVALS};

int NIN;        //number of input nodes
int S;

//loades any type of data
void load_data(int type){
    if(type==BOOL)          BoolData.clear();
    else if(type==REAL)     RealData.clear();
    else if(type==CLASS)    ClassData.clear();

    char tmp[SIZE], *p;
    while(gets(tmp)!=NULL){

        vector<int>     tInt;
        vector<double>  tDouble;
        tInt.push_back(1);
        tDouble.push_back(1.);

        p = strtok(tmp, DELIM);     //tokenize the line
        while(p != NULL){
            if(type == BOOL)    tInt.push_back( atoi(p) );
            else                tDouble.push_back( atof(p) );

            p = strtok(NULL, DELIM);
        }


        if(type==BOOL)          BoolData.push_back(tInt);
        else if(type==REAL)     RealData.push_back(tDouble);
        else if(type==CLASS)    ClassData.push_back(tDouble);
    }

    if(type == BOOL)        NIN = BoolData[0].size()-1;
    else if(type == REAL)   NIN = RealData[0].size()-1;
    else                    NIN = ClassData[0].size()-1;

    //printf("NIN: %d\n",NIN);

}

void randomWeight(int m){
    int i,j;
    for(i=0;i<NIN;i++){
        for(j=0;j<=m;j++){
            _L1W[i][j] = (rand()%101)/100. - 0.5;
        }
    }
    for(i=0;i<=m;i++)    _L2W[i] = (rand()%101)/100. - 0.5;
}

void copyWeight(int m){
    memcpy(L1W, _L1W, sizeof(_L1W));
    memcpy(L2W, _L2W, sizeof(_L2W));
}

double feedForward(int type, int s, int m){
    int i,j;
    for(j=0;j<=m;j++){
        oH[j] = 0.;
        for(i=0;i<NIN;i++){

            if(type == BOOL)        oH[j] += ( BoolData[s][i]*L1W[i][j] );
            else if(type == REAL)   oH[j] += ( RealData[s][i]*L1W[i][j] );
            else                    oH[j] += ( ClassData[s][i]*L1W[i][j] );
        }
    }

    for(j=0;j<=m;j++)    oH[j] = 1/( 1+pow(M_E, -oH[j]) );


    //output output
    oK = 0.;
    for(j=0;j<=m;j++){
        oK += ( oH[j]*L2W[j] );
    }
    oK = 1/(1+pow(M_E, -oK));

    return oK;
}

void backPropagate(int type, int sample, int m){
    //output node del
    if(type == BOOL)        delK = oK * (1-oK) * (BoolData[sample][NIN]-oK);
    else if(type == REAL)   delK = oK * (1-oK) * (RealData[sample][NIN]-oK);
    else                    delK = oK * (1-oK) * (ClassData[sample][NIN]-oK);


    //hidden node del
    for(int j=0;j<=m;j++){
        delH[j] = oH[j] * (1-oH[j]) * L2W[j] * delK;
    }
}

double ANN(int type, int itr, int m, double eta){

    int i,j,s,it;

    copyWeight(m);

    double lasterror,error;
    lasterror = 999999.;

    it=0;
    while(it < itr){

        error = 0.;

        for(s=0;s<S;s++){
            //printf("%g\n",L2W[0]);

            feedForward(type, s, m);

            backPropagate(type, s, m);

            //update weight
            for(j=0;j<=m;j++){
                for(i=0;i<NIN;i++){
                    if(type == BOOL)        L1W[i][j] += (eta * delH[j] * BoolData[s][i]);
                    else if(type == REAL)   L1W[i][j] += (eta * delH[j] * RealData[s][i]);
                    if(type == CLASS)       L1W[i][j] += (eta * delH[j] * ClassData[s][i]);
                }
            }

            for(j=0;j<=m;j++){
                L2W[j] += (eta * delK * oH[j]);
            }


            //update error
            if(type == BOOL)        error += ( (BoolData[s][NIN] - oK) * (BoolData[s][NIN] - oK) );
            else if(type == REAL)   error += ( (RealData[s][NIN] - oK) * (RealData[s][NIN] - oK) );
            else                    error += ( (ClassData[s][NIN] - oK) * (ClassData[s][NIN] - oK) );

        }

        error /= 2.;
        //printf("Error: %g\n",error);
        if(error > lasterror)   {it=itr; break;}
        lasterror = error;

        it++;
    }
    //printf("run %d iterations..\n",it);
    return lasterror;
}


double calc(int type, int m){
    double out,acc,std,mean;
    int i,cnt = 0;

    acc = mean = std = 0.;
    if(type == REAL){
        for(i=0;i<S;i++)    mean += RealData[i][NIN];
        mean /= S;
    }

    for(i=0;i<S;i++){
        out = feedForward(type, i, m);
        //printf("**%.2lf\n",out);
        if(type == BOOL){
            if( (out <= 0.2 && BoolData[i][NIN]==0) || (out>=0.8 && BoolData[i][NIN]==1) )    cnt++;
        }else if(type == CLASS){
            if( (out <= 0.2 && ClassData[i][NIN]==0) || (out>=0.8 && ClassData[i][NIN]==1) )    cnt++;
        }else{
            std += (out-mean) * (out-mean);
        }
    }

    if(type == REAL){
        std /= S;
        std = sqrt(std);
        //printf("STD     :  %.2lf%%\n",std);
        return std;
    }

    acc = (cnt*100.)/S;
    //printf("Accuracy:  %.2lf%%\n",acc);
    return acc;

}


void test_ANN(int type){
    int itr=100;
    double err, minerr=9999999., mItr, mEta, mM;

    for(int i=0;i<MM;i++){
        randomWeight(MV[i]);
        printf("M: %d\n",MV[i]);
        printf("-----\n\n");
        printf("\t%-8s%8s%8s%8s%8s\n","eta","10", "100", "1000", "10000");
        printf("\t%-8s%8s%8s%8s%8s\n","----","-------", "-------", "-------", "-------");
        for(double eta=0.1; eta<=0.9;eta+=0.2){
            printf("\t%-8.2lf",eta);
            for(int itr=10;itr<=10000;itr*=10){
                err = ANN(type, itr, MV[i], eta);

                //printf("%8.4lf",err);
                if(type == REAL)    printf("%8.4lf",calc(type, MV[i]));
                else                printf("%7.2lf%%",calc(type, MV[i]));


                if(err < minerr){
                    minerr = err;

                    mItr = itr;
                    mEta = eta;
                    mM = MV[i];
                }

            }
            cout << endl;
        }
        cout << endl;
    }

    printf("\n\nMinimum Error: %g\n",minerr);
    printf("M: %.2lf, eta: %.2lf, itr: %.2lf\n\n",mM, mEta, mItr);
}

int main()
{
    ::srand(time(NULL));

    IN1
    OUT1
    load_data(BOOL);
    S = BoolData.size();
    printf("Boolean Data loaded: %6d\n",S);
    test_ANN(BOOL);

/*
    randomWeight(40);
    ANN(BOOL, 1000, 40, .3);
    for(int i=0;i<S;i++)
        printf("%.0lf\n",feedForward(BOOL, i, 40));
    printf("\n%8.2lf%%\n\n",calc(BOOL, 40) );
*/

    IN2
    OUT2
    load_data(BOOL);
    S = BoolData.size();
    printf("Boolean Data loaded: %6d\n",S);
    test_ANN(BOOL);



    IN3
    OUT3
    load_data(REAL);
    S = RealData.size();
    printf("Real Data loaded   : %6d\n",S);
    test_ANN(REAL);


    IN4
    OUT4
    load_data(CLASS);
    S = ClassData.size();
    printf("Class Data loaded  : %6d\n",S);
    test_ANN(CLASS);

    /*
    randomWeight(40);
    ANN(CLASS, 1000000, 40, .3);
    for(int i=0;i<10;i++)
        printf("%.0lf\n",feedForward(CLASS, i, 40));
    */

	return 0;
}


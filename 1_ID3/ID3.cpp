/*
Title : ID3 Algorithm implementation
Date  : 19.10.2011
Auther: Ragib Ahsan, Dhiman Paul, Riyad Parvez
Course: CSE472, Assignment-1
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
#include <algorithm>
#include <iterator>
using namespace std;

#define MAXLINESIZE 100
#define ATTRSIZE 10
#define NUMATTR 9
#define YES 1
#define NO 0
#define TP 0
#define FP 1
#define TN 2
#define FN 3

#define EPS 1e-7
#define MAX(a,b) (a)>(b)?(a):(b)
#define MIN(a,b) (a)<(b)?(a):(b)

#define IN  freopen("data.csv","r",stdin);
#define OUT  freopen("out.txt","w",stdout);


vector<vector<int> > data;
vector<int> train_data;
vector<int> test_data;
vector<int> attributes;
set<int> total;
int target_attr;

//a typical node of the decision tree
struct node{
    int attrid;
    bool leaf;      //either leaf node or not
    node *child[ATTRSIZE];
    node() {leaf = true;}
    node(int id) {
        attrid = id;
        leaf = false;
        for(int i=0;i<ATTRSIZE;i++) child[i] = NULL;
    }
};

//returns the most common value of the data subset D
int most_common_value(vector<int> D)
{
    int i,p,len = D.size();
    for(i=0, p=0;i<len;i++){
        if(data[ D[i] ][ target_attr ] == 1)  p++;
    }
    int m = MAX(p,len-p);
    return (p==m)?1:0;
}

//returns the entropy of the data subset D with the attribute 'attr' having value 'val'
//'attr' and 'val' are optional arguments
//keep impurity=true for getting impurity instead of entropy
double entropy(bool impurity, vector<int> D, int attr=-1, int val=-1)
{
    int i,j,p,len = D.size();
    for(i=0,p=0;i<len;i++){
        if(attr != -1 && data[ D[i] ][ attr ] != val)   continue;
        if(data[ D[i] ][ target_attr ] == 1)  p++;
    }

    if(p == len || p == 0) return 0;

    double tot = (len) * 1.;

    if(impurity)    return (1 - (p/tot)*(p/tot) - ((len-p)/tot)*((len-p)/tot) );                //impurity
    return -( (p/tot)*( log(p/tot)/log(2) ) ) - ( ((len-p)/tot)*( log((len-p)/tot)/log(2) ) );  //entropy
}

//returns information gain(impurity=false) or misclassification impurity(impurity=true)
//with data subset 'D' and attribute 'attr'
double gain(bool impurity, vector<int> D, int attr)
{
    int vals[ATTRSIZE+1];
    int i,j,len = D.size();
    double e = entropy(impurity, D);

    memset(vals, 0, sizeof(vals));
    for(i=0;i<len;i++)  vals[ data[ D[i] ][ attr ] ]++;

    for(i=1;i<=ATTRSIZE;i++){
        if(vals[i] == 0)    continue;
        double en = entropy(impurity, D, attr, i);
        e -= ( vals[i]/(len*1.) ) * en;
    }

    return e;
}

//get the example instances from the data subset D where attribute 'attr' has value 'val'
vector<int> get_example(vector<int> D, int attr, int val)
{
    vector<int> temp;
    int i,len = D.size();
    for(i=0;i<len;i++){
        if(data[ D[i] ][ attr ] == val)   temp.push_back(D[i]);
    }
    return temp;
}

//create the ID3 decision tree from the data subset 'D'
//either information gain(impurity=false) or misclassification impurity(impurity=true) can be used
node *build_ID3_tree(bool impurity, vector<int> D)
{
    node *root;
    if(D.empty() || attributes.empty()){    //return the most common value when no attribute
        root = new node();
        root->attrid = most_common_value(D);
        return root;
    }

    double en = entropy(impurity, D);
    if( fabs(en-0.) <= EPS ){               //return the regular value when entropy zero
        root = new node();
        if(data[ D[0] ][ target_attr ] == 1)  root->attrid = YES;
        else  root->attrid = NO;
        return root;
    }

    int i,b=attributes[0];
    double max = 0.;
    for(i=0;i<attributes.size();i++){       //determine the attribute 'b' with maximum gain/impurity
        double temp = gain(impurity, D, attributes[i]);
        if(temp > max){
            max = temp;
            b = attributes[i];
        }
    }

    root = new node(b);     //create node with the best attribute

    vector<int>::iterator it = attributes.begin();
    while(it != attributes.end()){
        if(*it == b)    break;
        it++;
    }
    if(it != attributes.end())  attributes.erase(it);   //remove the attribue from attribute list

    for(i=1;i<=ATTRSIZE;i++){   //build child tree
        vector<int> dd = get_example(D,b,i);    //get data subset
        root->child[i-1] = build_ID3_tree(impurity, dd);
    }

    return root;    //return the root of the complete tree
}

//free the memory occupied by the tree
void free_ID3_tree(node *root)
{
    if(root == NULL)    return;
    if(root->leaf)  {free(root); return;}

    for(int i=0;i<ATTRSIZE;i++){
        free_ID3_tree(root->child[i]);
    }

    free(root);
}


//partition the data into 80% training data into 'train_data' vector
//and 20% test data into 'test_data' vector
void partition_data()
{
    train_data.clear();
    test_data.clear();

    int i,len = data.size();
    int r,n = ceil(len * 0.2);
    set<int> s;

    while(true){
        r = rand()%len;
        s.insert(r);
        if (s.size() == n)  break;
    }

    test_data = vector<int>(s.begin(), s.end());    //random 20% data
    set_difference(total.begin(), total.end(), s.begin(),s.end(),std::inserter(train_data, train_data.end()));  //rest 80% data
}

//initialize the attributes vector
void init_attributes()
{
    attributes.clear();
    for(int i=0;i<NUMATTR;i++){
        attributes.push_back(i);
    }

    target_attr =  NUMATTR;
}

//load data from file
void load_data()
{
    char line[MAXLINESIZE],*p;
    int i,v;
    vector<int> temp;

    if(!data.empty())   data.clear();

    while(gets(line)){
        p = strtok(line,", ");
        temp.clear();
        while(p!=NULL){
            v = atof(p);
            temp.push_back(v);
            p = strtok(NULL,", ");
        }
        data.push_back(temp);
    }

    if(data.empty())    return;

    int len = data.size();
    for(i=0;i<len;i++)  total.insert(i);
}

//test an 'example' instance and return decision for target attribute with the given tree rooted at 'root'
int test(node *root, vector<int> example){
    if(root == NULL) {printf("NULL tree!\n"); return -1;}
    while(!root->leaf){
        root = root->child[ example[root->attrid]-1 ];
    }
    return root->attrid;
}

//run test for all the test data and count true positive, false positive, true negative, false negative
//and add the counts to the corresponding reference arguments (i.e. tp, fp, tn, fn)
void analysis(node *root, int &tp, int &fp, int &tn, int &fn){
    int i,len = test_data.size();
    for(i=0;i<len;i++){
        vector<int> v = data[test_data[i]]; //an instance
        vector<int>::iterator it = v.end();
        it--;
        int val = *it;
        v.erase(it);                        //remvove the target attribute

        int valout = test(root, v);         //test

        if(val==1 && valout==1) tp++;
        else if(val==1) fn++;
        else if(valout==1)  fp++;
        else tn++;                          //count
    }
}

//print the analysis result
void print_analysis(int M[2][4])
{
    printf("            %8s  %8s\n","IG","MI");
    printf("            %s  %s\n","--------","--------");
    printf("%-10s: %8.3lf  %8.3lf\n", "True+", M[0][TP]/100., M[1][TP]/100.);
    printf("%-10s: %8.3lf  %8.3lf\n", "True-", M[0][TN]/100., M[1][TN]/100.);
    printf("%-10s: %8.3lf  %8.3lf\n", "False+", M[0][FP]/100., M[1][FP]/100.);
    printf("%-10s: %8.3lf  %8.3lf\n", "False-", M[0][FN]/100., M[1][FN]/100.);

    double a1 = ((M[0][TP]+M[0][TN])*1.)/(M[0][TP]+M[0][TN]+M[0][FP]+M[0][FN]);
    double a2 = ((M[1][TP]+M[1][TN])*1.)/(M[1][TP]+M[1][TN]+M[1][FP]+M[1][FN]);
    double p1 = (M[0][TP]*1.)/(M[0][TP]+M[0][FP]);
    double p2 = (M[1][TP]*1.)/(M[1][TP]+M[1][FP]);
    double r1 = (M[0][TP]*1.)/(M[0][TP]+M[0][FN]);
    double r2 = (M[1][TP]*1.)/(M[1][TP]+M[1][FN]);

    printf("%-10s: %7.3lf%%  %7.3lf%%\n","Accuracy",100*a1,100*a2);
    printf("%-10s: %7.3lf%%  %7.3lf%%\n","Precision",100*p1, 100*p2);
    printf("%-10s: %7.3lf%%  %7.3lf%%\n","Recall",100*r1, 100*r2);
    printf("%-10s: %7.3lf%%  %7.3lf%%\n","F-Measure",100*(2.*p1*r1)/(p1+r1), 100*(2.*p2*r2)/(p2+r2));
    printf("%-10s: %7.3lf%%  %7.3lf%%\n","G-Mean",100*sqrt( (M[0][TP]*M[0][TN]*1.)/((M[0][TP]+M[0][FN])*(M[0][TN]+M[0][FP])) ), 100*sqrt( (M[1][TP]*M[1][TN]*1.)/((M[1][TP]+M[1][FN])*(M[1][TN]+M[1][FP])) ));
}

int main()
{
    IN
    OUT
    load_data();
    int i,j;
    int M[2][4];
    node *root;

    memset(M,0,sizeof(M));

    srand(time(NULL));
    for(j=0;j<2;j++){
        for(i=0;i<100;i++){
            init_attributes();
            partition_data();
            root = build_ID3_tree(j, train_data);
            analysis(root, M[j][TP], M[j][FP], M[j][TN], M[j][FN]);
            free_ID3_tree(root);
        }
    }

    print_analysis(M);

    return 0;
}

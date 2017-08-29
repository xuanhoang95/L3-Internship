#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <unsupported/Eigen/MPRealSupport>

using namespace std;
using namespace mpfr;
using namespace Eigen;

typedef Matrix<mpreal,Dynamic,Dynamic>  Mat;

#define VM vector<Mat>
#define VVM vector<vector<Mat>>
#define FOR(x) for(int x=0;x<2;x++)

vector <tuple<int,int,int,int>> forbidden;

//Sort comparison
bool compare(pair <int,mpreal> a, pair <int,mpreal> b){
    return a.second>b.second;
}

//omega(a,b,c,d) (6)
mpreal isValid(int a, int b, int c, int d, string s){
    if(s=="hs"){
        if((a&&b)||(c&&d)||(a&&c)||(b&&d)){
            return 0;
        }
        else{
            return 1;
        }
    }
    else if(s=="nak"){
        if(a+b+c+d==0 || a+b+c+d==1){
            return 1;
        }
        else{
            return 0;
        }
    }
    else if(s=="even"){
        if(a+b+c+d==2){
            return 0;
        }
        else{
            return 1;
        }
    }
    else{
        if(find(forbidden.begin(),forbidden.end(),make_tuple(a,b,c,d))!=forbidden.end()){
            return 0;
        }
        else{
            return 1;
        }
    }
}

//Expand A and F (22)
pair <VM,VVM> expand(VM A, VVM F, string s){
    int p = 2*A[0].rows();
    Mat Al[2][2][2][2];
    Mat Fl[2][2][2][2];
    FOR(c){
        FOR(a){
            FOR(d){
                FOR(b){
                    Al[c][d][a][b]=isValid(a,b,c,d,s)*F[d][b]*A[b]*F[b][a];
                    Fl[d][c][b][a]=isValid(a,b,c,d,s)*F[b][a];
                }
            }
        }
    }
    VM resultA(2);
    FOR(i){
        resultA[i].resize(p,p);
        resultA[i]<< Al[i][0][0][0]+Al[i][0][0][1],Al[i][0][1][0]+Al[i][0][1][1],Al[i][1][0][0]+Al[i][1][0][1],Al[i][1][1][0]+Al[i][1][1][1];
    }
    VVM resultF(2,VM(2));
    FOR(i){
        FOR(j){
            resultF[i][j].resize(p,p);
            resultF[i][j]<<Fl[i][j][0][0],Fl[i][j][0][1],Fl[i][j][1][0],Fl[i][j][1][1];
        }
    }
    pair <VM,VVM> result = make_pair(resultA,resultF);
    return result;
}

//Increase n
mpreal increase(VM P, VM Pl, VM E, VM El){
    mpreal res=0;
    FOR(i){
        res= max((E[i]-El[i]).lpNorm<Infinity>(),res);
        res= max((P[i]-Pl[i]).lpNorm<Infinity>(),res);
        //cout<<"****"<<endl;
        //cout<<P[i]<<endl<<endl;
        //cout<<Pl[i]<<endl<<endl;
    }
    return res;
}

//Reduce A and F (23)
tuple <VM,VVM,VM,VM,VM> reduce(VM A, VVM F, VM tl, int n){
    int p=A[0].rows();
    VM P(2);
    VM E(2);
    VM t(2);
    FOR(i){
        SelfAdjointEigenSolver<Mat> eigensolver(A[i]);
        E[i] = eigensolver.eigenvalues();
        //cout << E[i]<<endl<<endl;
        vector < pair <int,mpreal> > lambda;
        for(int x=0;x<p;x++){
            pair <int,mpreal> el=make_pair(x,abs(E[i](x,0)));
            lambda.push_back(el);
        }
        sort(lambda.begin(),lambda.end(),compare);
        t[i] = eigensolver.eigenvectors();
        //if(i==0)cout<<t[i]<<endl<<endl;
        //if(i==0)cout<<tl[i]<<endl<<endl;
        if(t[i].size()==tl[i].size()){
            for(int j=0;j<t[i].cols();j++){
                bool b=true;
                for(int k=0;k<t[i].rows() && b;k++){
                    if(t[i](k,j)*tl[i](k,j)>=0){
                        b=false;
                    }
                }
                if(b){
                    //cout<<"******"<<endl;
                    //cout<<tl[i].col(j)<<endl<<endl;
                    //cout<<t[i].col(j)<<endl<<endl;
                    t[i].col(j)=-t[i].col(j);
                }
            }
        }
        P[i]= Mat::Zero(p,n);
        for(int x=0;x<n;x++){
            P[i].col(x)=t[i].col(lambda[x].first);
        }
    }
    //cout<<"****"<<endl;
    //cout<<t[0]<<endl<<endl;
    //cout<<t[1]<<endl<<endl;
    FOR(i){
        A[i]=P[i].transpose()*A[i]*P[i];
        FOR(j){
            F[i][j]=P[i].transpose()*F[i][j]*P[j];
        }
    }
    mpreal ca = A[0](0,0);
    mpreal cf = F[0][0](0,0);
    FOR(i){
        A[i]/=ca;
        FOR(j){
            F[i][j]/=cf;
        }
    }
    tuple <VM,VVM,VM,VM,VM> result = make_tuple(A,F,P,E,t);
    return result;
}

//Power method
mpreal lowerBound(VM A, VVM F, int powprec, string s){
    VM X(2);
    VVM Y(2,VM(2));
    int n = A[0].rows();
    // (21)
    FOR(a){
        X[a]=A[a]*A[a];
        Mat ran = Mat::Random(n,n);
        X[a]+=ran/ran.squaredNorm(); //slight perturbation
        FOR(b){
            Y[a][b]=A[a]*F[a][b]*A[b];
        }
    }
    // (19) and (20)
    VM Xl(2);
    VVM Yl(2,VM(2));
    int p=0;
    mpreal prec=1;
    while(prec>=pow(10,-powprec)){
        if(p%2){
            prec = 0;
            FOR(a){
                mpreal dif;
                if(X[a].lpNorm<Infinity>()>0){
                    dif=Xl[a].squaredNorm()/X[a].squaredNorm();
                    X[a]=F[a][1-a]*Xl[1-a]*F[1-a][a]+F[a][a]*Xl[a]*F[a][a];
                    dif-=X[a].squaredNorm()/Xl[a].squaredNorm();
                    prec+= dif>0 ? dif : -dif;
                }
                FOR(b){
                    if(Y[a][b].lpNorm<Infinity>()>0){
                        dif=Yl[a][b].squaredNorm()/Y[a][b].squaredNorm();
                        Y[a][b]= Mat::Zero(n,n);
                        FOR(c){
                            FOR(d){
                                Y[a][b]+=isValid(a,b,c,d,s)*F[a][c]*Yl[c][d]*F[d][b];
                            }
                        }
                        dif-=Y[a][b].squaredNorm()/Yl[a][b].squaredNorm();
                        prec+= dif>0 ? dif : -dif;
                    }
                }
            }
        }
        else{
            FOR(a){
                Xl[a]=F[a][1-a]*X[1-a]*F[1-a][a]+F[a][a]*X[a]*F[a][a];
                FOR(b){
                    Yl[a][b]= Mat::Zero(n,n);
                    FOR(c){
                        FOR(d){
                            Yl[a][b]+=isValid(a,b,c,d,s)*F[a][c]*Y[c][d]*F[d][b];
                        }
                    }
                }
            }
        }
        p++;
    }
    cout<<"power method iteration : "<<p<<endl;
    mpreal eta=0, epsilon =0;
    mpreal eta_c=0, epsilon_c =0;
    FOR(i){
        if(X[i].lpNorm<Infinity>()>0){
            epsilon+= X[i].squaredNorm()/Xl[i].squaredNorm();
            epsilon_c+= 1;
        }
        FOR(j){
            if(Y[i][j].lpNorm<Infinity>()>0){
                eta+= Y[i][j].squaredNorm()/Yl[i][j].squaredNorm();
                eta_c+= 1;
            }
        }
    }
    epsilon = sqrt(epsilon/epsilon_c);
    eta = sqrt(eta/eta_c);
    return eta/epsilon;
}

//Estimate (24)
mpreal estimate(VM A, VVM F, string s){
    VM A2(2);
    A2[0]=A[0]*A[0];
    A2[1]=A[1]*A[1];
    mpreal Z = (A2[0]*A2[0]+A2[1]*A2[1]).trace();
    mpreal den = 0;
    mpreal num = 0;
    FOR(a){
        FOR(b){
            den+= (A2[a]*F[a][b]*A2[b]*F[b][a]).trace();
            FOR(c){
                FOR(d){
                    num+= (isValid(a,b,c,d,s)*A[a]*F[a][c]*A[c]*F[c][d]*A[d]*F[d][b]*A[b]*F[b][a]).trace();
                }
            }
        }
    }
    den*=den;
    return Z*num/den;
}

//Graph
void graph(Mat e, int speed, mpreal dif){
    cout<<speed;
    for(int i=e.rows()-1;i>=0;i--){
        cout<<' '<<abs(e(i,0));
    }
    cout<<' '<<dif<<endl;
}

//test
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    // Reading input
    int rsize, prec, msize;
    string s;
    cin >> rsize >> prec >> msize >> s;
    int a,b,c,d;
    while(cin>>a>>b>>c>>d){
        forbidden.push_back(make_tuple(a,b,c,d));
    }
    // Set default precision
    mpreal::set_default_prec(rsize);
    // Algorithm
    VM A(2);
    VVM F(2,VM(2));
    VM P(2);
    VM E(2);
    FOR(a){
        P[a].resize(1,1);
        E[a].resize(1,1);
        A[a].resize(1,1);
        A[a]<<1;
        FOR(b){
            F[a][b].resize(1,1);
            F[a][b]<<1;
        }
    }
    VM Pl(2);
    VM El(2);
    VM tl(2);
    int n=1;
    int convspeed=0;
    mpreal dif;
    pair < VM, VVM > result;
    while(n<msize){
        result = expand(A,F,s);
        A=result.first;
        F=result.second;
        convspeed++;
        if(P[0].size()==Pl[0].size()){
            dif=increase(P,Pl,E,El);
            //cout<<"****"<<endl<<endl<<E[0]<<endl<<endl<<E[1]<<endl<<endl;
            if(dif<pow(10,-prec)){
                cout<<n<<" : "<<convspeed<<endl;
                n++;
                convspeed=0;
            }
        }
        Pl = P;
        El = E;
        //cout<<P[0].size()<<' '<<Pl[0].size()<<endl;
        //if(n==20 && convspeed>2) graph(E[0],convspeed, dif);
        tie(A,F,P,E,tl) = reduce(A,F,tl,n);
    }
    mpreal lB = lowerBound(A,F,prec,s);
    mpreal kappa =estimate(A,F,s);
    cout<<"Lower bound : "<<setprecision(prec)<<lB<<endl;
    cout<<"Estimate : "<<setprecision(prec)<<kappa<<endl;
}

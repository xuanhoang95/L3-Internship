#include <iostream>
#include <iomanip>
#include <vector>
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
    if(s=="rwim"){
        if((a&&b) || (a&&d) || (c&&b) || (c&&d)){
            return 0;
        }
        else{
            return 1;
        }
    }
    else if(s=="hh"){
        if((a&&b) || (a&&c) || (b&&d) || (c&&d) || (a&&d)){
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
tuple <VM, VM, VVM, VVM> expand(VM A, VM B, VVM F, VVM G, string s){
    int p = 2*A[0].rows();
    Mat Al[2][2][2][2];
    Mat Bl[2][2][2][2];
    Mat Fl[2][2][2][2];
    Mat Gl[2][2][2][2];
    FOR(a){
        FOR(b){
            FOR(c){
                FOR(d){
                    Al[a][b][c][d]=isValid(a,b,c,d,s)*G[b][a]*A[a]*F[a][c];
                    Bl[a][b][c][d]=isValid(a,b,c,d,s)*F[d][b]*B[b]*G[b][a];
                    Fl[a][b][c][d]=isValid(a,b,c,d,s)*F[d][b];
                    Gl[a][b][c][d]=isValid(a,b,c,d,s)*G[b][a];
                }
            }
        }
    }
    VM resultA(2);
    VM resultB(2);
    FOR(i){
        resultA[i].resize(p,p);
        resultA[i]<< Al[0][0][0][i]+Al[1][0][0][i],Al[0][0][1][i]+Al[1][0][1][i],Al[0][1][0][i]+Al[1][1][0][i],Al[0][1][1][i]+Al[1][1][1][i];
        resultB[i].resize(p,p);
        resultB[i]<< Bl[0][0][i][0]+Bl[0][1][i][0],Bl[1][0][i][0]+Bl[1][1][i][0],Bl[0][0][i][1]+Bl[0][1][i][1],Bl[1][0][i][1]+Bl[1][0][i][1];
    }
    VVM resultF(2,VM(2));
    VVM resultG(2,VM(2));
    FOR(i){
        FOR(j){
            resultF[i][j].resize(p,p);
            resultF[i][j]<<Fl[j][0][i][0],Fl[j][1][i][0],Fl[j][0][i][1],Fl[j][1][i][1];
            resultG[i][j].resize(p,p);
            resultG[i][j]<<Gl[0][0][j][i],Gl[1][0][j][i],Gl[0][1][j][i],Gl[1][1][j][i];
        }
    }
    tuple <VM, VM, VVM, VVM> result = make_tuple(resultA, resultB, resultF, resultG);
    return result;
}

//Increase n
mpreal increase(VVM P, VVM Pl, VVM E, VVM El){
    mpreal res=0;
    FOR(i){
        FOR(j){
            res= max((E[i][j]-El[i][j]).lpNorm<Infinity>(),res);
            res= max((P[i][j]-Pl[i][j]).lpNorm<Infinity>(),res);
        }
    }
    return res;
}

//Reduce A and F (23)
tuple <VM, VM, VVM, VVM, VVM, VVM, VVM> reduce(VM A, VM B, VVM F, VVM G, VVM tl, int n){
    int p=A[0].rows();
    VVM P(2,VM(2));
    VVM E(2,VM(2));
    VVM t(2,VM(2));
    FOR(i){
        SelfAdjointEigenSolver<Mat> eigensolver(A[i]*B[i]);
        E[0][i] = eigensolver.eigenvalues();
        vector < pair <int,mpreal> > lambda;
        for(int x=0;x<p;x++){
            pair <int,mpreal> el=make_pair(x,abs(E[0][i](x,0)));
            lambda.push_back(el);
        }
        sort(lambda.begin(),lambda.end(),compare);
        t[0][i] = eigensolver.eigenvectors();
        if(t[0][i].size()==tl[0][i].size()){
            for(int j=0;j<t[0][i].cols();j++){
                bool b=true;
                if(i){
                    /*for(int k=0;k<t[0][i].rows() && b;k++){
                        if(t[0][i](k,j)*t[0][0](k,j)>=0){
                            b=false;
                        }
                    }
                    if(b){
                        //cout<<"******"<<endl;
                        //cout<<tl[i].col(j)<<endl<<endl;
                        //cout<<t[i].col(j)<<endl<<endl;
                        t[0][i].col(j)=-t[0][i].col(j);
                    }*/
                    for(int k=0;k<t[0][i].rows() && b;k++){
                        if(t[0][i](k,j)*tl[0][i](k,j)>=0){
                            b=false;
                        }
                    }
                    if(b){
                        //cout<<"******"<<endl;
                        //cout<<tl[i].col(j)<<endl<<endl;
                        //cout<<t[i].col(j)<<endl<<endl;
                        t[0][i].col(j)=-t[0][i].col(j);
                    }
                }
                else{
                    for(int k=0;k<t[0][i].rows() && b;k++){
                        if(t[0][i](k,j)*tl[0][i](k,j)>=0){
                            b=false;
                        }
                    }
                    if(b){
                        //cout<<"******"<<endl;
                        //cout<<tl[i].col(j)<<endl<<endl;
                        //cout<<t[i].col(j)<<endl<<endl;
                        t[0][i].col(j)=-t[0][i].col(j);
                    }
                }
            }
        }
        P[0][i]= Mat::Zero(p,n);
        for(int x=0;x<n;x++){
            P[0][i].col(x)=t[0][i].col(lambda[x].first);
        }
    }
    FOR(i){
        SelfAdjointEigenSolver<Mat> eigensolver(B[i]*A[i]);
        E[1][i] = eigensolver.eigenvalues();
        vector < pair <int,mpreal> > lambda;
        for(int x=0;x<p;x++){
            pair <int,mpreal> el=make_pair(x,abs(E[1][i](x,0)));
            lambda.push_back(el);
        }
        sort(lambda.begin(),lambda.end(),compare);
        t[1][i] = eigensolver.eigenvectors();
        if(t[1][i].size()==tl[1][i].size()){
            for(int j=0;j<t[1][i].cols();j++){
                bool b=true;
                if(i){
                    /*for(int k=0;k<t[1][i].rows() && b;k++){
                        if(t[1][i](k,j)*t[1][0](k,j)>=0){
                            b=false;
                        }
                    }
                    if(b){
                        //cout<<"******"<<endl;
                        //cout<<tl[i].col(j)<<endl<<endl;
                        //cout<<t[i].col(j)<<endl<<endl;
                        t[1][i].col(j)=-t[1][i].col(j);
                    }*/
                    for(int k=0;k<t[1][i].rows() && b;k++){
                        if(t[1][i](k,j)*tl[1][i](k,j)>=0){
                            b=false;
                        }
                    }
                    if(b){
                        //cout<<"******"<<endl;
                        //cout<<tl[i].col(j)<<endl<<endl;
                        //cout<<t[i].col(j)<<endl<<endl;
                        t[1][i].col(j)=-t[1][i].col(j);
                    }
                }
                else{
                    for(int k=0;k<t[1][i].rows() && b;k++){
                        if(t[1][i](k,j)*tl[1][i](k,j)>=0){
                            b=false;
                        }
                    }
                    if(b){
                        //cout<<"******"<<endl;
                        //cout<<tl[i].col(j)<<endl<<endl;
                        //cout<<t[i].col(j)<<endl<<endl;
                        t[1][i].col(j)=-t[1][i].col(j);
                    }
                }
            }
        }
        P[1][i]= Mat::Zero(p,n);
        for(int x=0;x<n;x++){
            P[1][i].col(x)=t[1][i].col(lambda[x].first);
        }
    }
    FOR(i){
        A[i]=P[0][i].transpose()*A[i]*P[1][i];
        B[i]=P[1][i].transpose()*B[i]*P[0][i];
        FOR(j){
            F[i][j]=P[1][i].transpose()*F[i][j]*P[1][j];
            G[i][j]=P[0][i].transpose()*G[i][j]*P[0][j];
        }
    }
    //display(A,F);
    mpreal ca = A[0](0,0);
    mpreal cf = F[0][0](0,0);
    FOR(i){
        A[i]/=ca;
        B[i]/=ca;
        FOR(j){
            F[i][j]/=cf;
            G[i][j]/=cf;
        }
    }
    tuple <VM, VM, VVM, VVM, VVM, VVM, VVM> result = make_tuple(A,B,F,G,P,E,t);
    return result;
}

//Power method
mpreal lowerBound(VM A, VM B, VVM F, VVM G, int powprec, string s){
    VM X(2);
    VVM Y(2,VM(2));
    int n = A[0].rows();
    // (21)
    FOR(a){
        X[a]=A[a]*B[a];
        Mat ran = Mat::Random(n,n);
        X[a]+=ran/ran.squaredNorm(); //slight perturbation
        FOR(b){
            Y[a][b]=A[a]*G[a][b]*B[b];
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
    VM B(2);
    VVM F(2,VM(2));
    VVM G(2,VM(2));
    VVM P(2,VM(2));
    VVM E(2,VM(2));
    FOR(a){
        A[a].resize(1,1);
        A[a]<<1;
        B[a].resize(1,1);
        B[a]<<1;
        FOR(b){
            P[a][b].resize(1,1);
            E[a][b].resize(1,1);
            F[a][b].resize(1,1);
            F[a][b]<<1;
            G[a][b].resize(1,1);
            G[a][b]<<1;
        }
    }
    VVM Pl(2,VM(2));
    VVM El(2,VM(2));
    VVM tl(2,VM(2));
    int n=1;
    int convspeed=0;
    mpreal dif;
    while(n<msize){
        tie(A,B,F,G)=expand(A,B,F,G,s);
        convspeed++;
        if(P[0][0].size()==Pl[0][0].size()){
            dif=increase(P,Pl,E,El);
            //if (n==8) cout<<"******"<<endl<<endl<<E[0][0]<<endl<<endl<<E[1][1]<<endl<<endl;
            if(dif<pow(10,-prec)){
                cout<<n<<" : "<<convspeed<<endl;
                n++;
                convspeed=0;
            }
        }
        Pl = P;
        El = E;
        tie(A,B,F,G,P,E,tl) = reduce(A,B,F,G,tl,n);
        //display(A,F);
    }
    mpreal lB = lowerBound(A,B,F,G,prec,s);
    cout<<"Lower bound : "<<setprecision(prec)<<lB<<endl;
}

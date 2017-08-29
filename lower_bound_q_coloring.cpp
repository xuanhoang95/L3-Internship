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
#define FOR(x,y) for(int x=0;x<y;x++)

//Sort comparison
bool compare(pair <int,mpreal> a, pair <int,mpreal> b){
    return a.second>b.second;
}

//omega(a,b,c,d) (6)
mpreal isValid(int a, int b, int c, int d){
    if(a==b || a==c || b==d || c==d){
        return 0;
    }
    else{
        return 1;
    }
}

//Expand A and F (22)
tuple <VM,VVM> expand(VM A, VVM F, int color){
    int p = A[0].rows();
    VM Al(color);
    VVM Fl(color,VM(color));
    FOR(c,color){
        Al[c] = Mat::Zero(p*color,p*color);
        FOR(d,color){
            Fl[d][c] = Mat::Zero(p*color,p*color);
            FOR(a,color){
                FOR(b,color){
                    Al[c].block(d*p,a*p,p,p)+=isValid(a,b,c,d)*F[d][b]*A[b]*F[b][a];
                    Fl[d][c].block(b*p,a*p,p,p)+=isValid(a,b,c,d)*F[b][a];
                }
            }
        }
    }
    //cout<<Fl[0][0]<<endl;
    tuple <VM,VVM> result = make_tuple(Al,Fl);
    return result;
}

//Increase n
mpreal increase(VM P, VM Pl, VM E, VM El, int color){
    mpreal res=0;
    FOR(i,color){
        res= max((E[i]-El[i]).lpNorm<Infinity>(),res);
        mpreal temp=0;
        for(int j=0;j<P[i].cols();j++){
            temp=max(temp,min((P[i].col(j)+Pl[i].col(j)).lpNorm<Infinity>(),(P[i].col(j)-Pl[i].col(j)).lpNorm<Infinity>()));
        }
        res=max(temp,res);
    }
    return res;
}

//Reduce A and F (23)
tuple <VM,VVM,VM,VM> reduce(VM A, VVM F, int n, int color){
    int p=A[0].rows();
    VM P(color);
    VM E(color);
    FOR(i,color){
        SelfAdjointEigenSolver<Mat> eigensolver(A[i]);
        E[i] = eigensolver.eigenvalues();
        vector < pair <int,mpreal> > lambda;
        for(int x=0;x<p;x++){
            pair <int,mpreal> el=make_pair(x,abs(E[i](x,0)));
            lambda.push_back(el);
        }
        sort(lambda.begin(),lambda.end(),compare);
        Mat t = eigensolver.eigenvectors();
        P[i]= Mat::Zero(p,n);
        for(int x=0;x<n;x++){
            P[i].col(x)=t.col(lambda[x].first);
        }
    }
    FOR(i,color){
        A[i]=P[i].transpose()*A[i]*P[i];
        FOR(j,color){
            F[i][j]=P[i].transpose()*F[i][j]*P[j];
        }
    }
    //display(A,F);
    mpreal ca = A[0](0,0); //cout<<ca<<endl;
    mpreal cf = F[1][0](0,0); //cout<<cf<<endl;
    FOR(i,color){
        A[i]/=ca;
        FOR(j,color){
            F[i][j]/=cf;
        }
    }
    tuple <VM,VVM,VM,VM> result = make_tuple(A,F,P,E);
    return result;
}

//Power method
mpreal lowerBound(VM A, VVM F, int powprec, int color){
    VM X(color);
    VVM Y(color,VM(color));
    int n = A[0].rows();
    // (21)
    FOR(a,color){
        X[a]=A[a]*A[a];
        Mat ran = Mat::Random(n,n);
        X[a]+=ran/ran.squaredNorm(); //slight perturbation
        FOR(b,color){
            Y[a][b]=A[a]*F[a][b]*A[b];
        }
    }
    // (19) and (20)
    VM Xl(color);
    VVM Yl(color,VM(color));
    int p=0;
    mpreal prec=1;
    while(prec>=pow(10,-powprec)){
        if(p%2){
            prec = 0;
            FOR(a,color){
                mpreal dif;
                if(X[a].lpNorm<Infinity>()>0){
                    dif=Xl[a].squaredNorm()/X[a].squaredNorm();
                    X[a] = Mat::Zero(n,n);
                    FOR(b,color){
                        X[a]+=F[a][b]*Xl[b]*F[b][a];
                    }
                    dif-=X[a].squaredNorm()/Xl[a].squaredNorm();
                    prec+= dif>0 ? dif : -dif;
                }
                FOR(b,color){
                    if(Y[a][b].lpNorm<Infinity>()>0){
                        dif=Yl[a][b].squaredNorm()/Y[a][b].squaredNorm();
                        Y[a][b] = Mat::Zero(n,n);
                        FOR(c,color){
                            FOR(d,color){
                                Y[a][b]+=isValid(a,b,c,d)*F[a][c]*Yl[c][d]*F[d][b];
                            }
                        }
                        dif-=Y[a][b].squaredNorm()/Yl[a][b].squaredNorm();
                        prec+= dif>0 ? dif : -dif;
                    }
                }
            }
        }
        else{
            FOR(a,color){
                Xl[a] = Mat::Zero(n,n);
                FOR(b,color){
                    Xl[a]+=F[a][b]*X[b]*F[b][a];
                    Yl[a][b]= Mat::Zero(n,n);
                    FOR(c,color){
                        FOR(d,color){
                            Yl[a][b]+=isValid(a,b,c,d)*F[a][c]*Y[c][d]*F[d][b];
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
    FOR(i,color){
        if(X[i].lpNorm<Infinity>()>0){
            epsilon+= X[i].squaredNorm()/Xl[i].squaredNorm();
            epsilon_c+= 1;
        }
        FOR(j,color){
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
    int rsize, prec, msize, color;
    cin >> rsize >> prec >> msize >> color;
    // Set default precision
    mpreal::set_default_prec(rsize);
    // Algorithm
    VM A(color);
    VVM F(color,VM(color));
    VM P(color);
    VM E(color);
    FOR(a,color){
        P[a].resize(1,1);
        E[a].resize(1,1);
        A[a].resize(1,1);
        A[a]<<1;
        FOR(b,color){
            F[a][b].resize(1,1);
            F[a][b]<<1;
        }
    }
    VM Pl(color);
    VM El(color);
    int n=1;
    int convspeed=0;
    mpreal dif;
    while(n<msize){
        tie(A,F)= expand(A,F,color);
        convspeed++;
        if(P[0].size()==Pl[0].size()){
            dif=increase(P,Pl,E,El,color);
            //cout<<dif<<endl;
            if(dif<pow(10,-prec)){
                cout<<n<<" : "<<convspeed<<endl;
                n++;
                convspeed=0;
            }
        }
        Pl = P;
        El = E;
        tie(A,F,P,E)= reduce(A,F,n,color);
    }
    mpreal lB = lowerBound(A,F,prec,color);
    cout<<"Lower bound : "<<setprecision(prec)<<lB<<endl;
}

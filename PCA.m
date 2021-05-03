%%
clc;       % 清除command window
clear      % 清除workspace
close all  % 關閉所有figure

%% 讀取.txt資料
dataSet = load('iris.txt');
rawData = dataSet(:,1:4);    % 原始資料，75筆資料 x 4個特徵
label   = dataSet(:,5);      % 75筆資料所對應的標籤
kk = 3; %knn k=3


%%計算特徵值與特徵向量
data = rawData';
mu = mean(data,2);
d = data - repmat(mu,1,150);
sigma = (d*d')*(1/(150-1));

[vector,values]=eig(sigma);%%計算特徵值特徵向量

V = orth(vector); %%特徵向量正規化

m = [3,4]; %%選取特徵
mi = [1:2];%%特徵選取範圍
k = size(m,2);
Vm = V(m,1:4);
x = data - mu;
y = Vm * x;
y = y';

trainset = [y(  1: 25,mi);...
          y( 51: 75,mi);...
          y(101:125,mi);]; 
          % 選取每類別前半，合併為training set

testset = [y( 26: 50,mi);...
          y( 76:100,mi);...
         y(126:150,mi)]; 
          % 選取每類別後半，合併為test set
          
error1 = 0;          
          
[testm,~]=size(testset);
[trainm,trainn]=size(trainset);

distancev=zeros(trainm,1);%每個測試點與訓練及的歐式距離量
for i=1:testm   
    for j=1:trainm
        distancev(j)=0;
        for k=1:k
            distancev(j)=distancev(j)+(testset(i,k)-trainset(j,k))^2; 
        end
        distancev(j)=sqrt(distancev(j));%.歐式距離
    end
    [val,index] = sort(distancev,'ascend');
     
    
    M = mode(label(index(1:kk)));
    
    

    
   
    if M ~= label(i,end)
        error1=error1+1;
    end

end

CR1=1-error1/testm;
fprintf('平均分類率CR1 = %2.4f%%\n', CR1*100)

%%
trainset = [y( 26: 50,mi);...
          y( 76:100,mi);...
         y(126:150,mi)]; 
          % 選取每類別後半，合併為test set
          
testset= [y(  1: 25,mi);...
          y( 51: 75,mi);...
          y(101:125,mi);]; 
          % 選取每類別前半，合併為training set
          
[testm,~]=size(testset);
[trainm,trainn]=size(trainset);

error2 = 0;

distancev=zeros(trainm,1);%每個測試點與訓練及的歐式距離量
for i=1:testm   
    for j=1:trainm
        distancev(j)=0;
        for k=1:k
            distancev(j)=distancev(j)+(testset(i,k)-trainset(j,k))^2; 
        end
        distancev(j)=sqrt(distancev(j));%.歐式距離
    end
    [val,index] = sort(distancev,'ascend');
     
    
    M = mode(label(index(1:kk)));
    
    

    
   
    if M ~= label(i,end)
        error2=error2+1;
    end

end

CR2=1-error2/testm;
fprintf('平均分類率CR2 = %2.4f%%\n', CR2*100)


CR = (CR1+CR2)/2;
fprintf('平均分類率CR = %2.4f%%\n', CR*100)


%%   前兩大特徵散佈圖
figure; % 開啟新的繪圖空間

plot(y(  1: 50,1),y(  1: 50,2),'ro',...
     y( 51:100,1),y( 51:100,2),'go',...
     y(101:150,1),y(101:150,2),'bo');   
     % 以plot繪圖指令分別畫出class1~3之第一與第二特徵。

title('Scatter Plot');                              % 圖名稱
legend('class1', 'class2', 'class3');               % 類別標號說明
xlabel('Feature3');                                 % 特徵標號註解
ylabel('Feature4');                                 % 特徵標號註解




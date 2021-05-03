%%
clc;       % �M��command window
clear      % �M��workspace
close all  % �����Ҧ�figure

%% Ū��.txt���
dataSet = load('iris.txt');
rawData = dataSet(:,1:4);    % ��l��ơA75����� x 4�ӯS�x
label   = dataSet(:,5);      % 75����Ʃҹ���������
kk = 3; %knn k=3


%%�p��S�x�ȻP�S�x�V�q
data = rawData';
mu = mean(data,2);
d = data - repmat(mu,1,150);
sigma = (d*d')*(1/(150-1));

[vector,values]=eig(sigma);%%�p��S�x�ȯS�x�V�q

V = orth(vector); %%�S�x�V�q���W��

m = [3,4]; %%����S�x
mi = [1:2];%%�S�x����d��
k = size(m,2);
Vm = V(m,1:4);
x = data - mu;
y = Vm * x;
y = y';

trainset = [y(  1: 25,mi);...
          y( 51: 75,mi);...
          y(101:125,mi);]; 
          % ����C���O�e�b�A�X�֬�training set

testset = [y( 26: 50,mi);...
          y( 76:100,mi);...
         y(126:150,mi)]; 
          % ����C���O��b�A�X�֬�test set
          
error1 = 0;          
          
[testm,~]=size(testset);
[trainm,trainn]=size(trainset);

distancev=zeros(trainm,1);%�C�Ӵ����I�P�V�m�Ϊ��ڦ��Z���q
for i=1:testm   
    for j=1:trainm
        distancev(j)=0;
        for k=1:k
            distancev(j)=distancev(j)+(testset(i,k)-trainset(j,k))^2; 
        end
        distancev(j)=sqrt(distancev(j));%.�ڦ��Z��
    end
    [val,index] = sort(distancev,'ascend');
     
    
    M = mode(label(index(1:kk)));
    
    

    
   
    if M ~= label(i,end)
        error1=error1+1;
    end

end

CR1=1-error1/testm;
fprintf('���������vCR1 = %2.4f%%\n', CR1*100)

%%
trainset = [y( 26: 50,mi);...
          y( 76:100,mi);...
         y(126:150,mi)]; 
          % ����C���O��b�A�X�֬�test set
          
testset= [y(  1: 25,mi);...
          y( 51: 75,mi);...
          y(101:125,mi);]; 
          % ����C���O�e�b�A�X�֬�training set
          
[testm,~]=size(testset);
[trainm,trainn]=size(trainset);

error2 = 0;

distancev=zeros(trainm,1);%�C�Ӵ����I�P�V�m�Ϊ��ڦ��Z���q
for i=1:testm   
    for j=1:trainm
        distancev(j)=0;
        for k=1:k
            distancev(j)=distancev(j)+(testset(i,k)-trainset(j,k))^2; 
        end
        distancev(j)=sqrt(distancev(j));%.�ڦ��Z��
    end
    [val,index] = sort(distancev,'ascend');
     
    
    M = mode(label(index(1:kk)));
    
    

    
   
    if M ~= label(i,end)
        error2=error2+1;
    end

end

CR2=1-error2/testm;
fprintf('���������vCR2 = %2.4f%%\n', CR2*100)


CR = (CR1+CR2)/2;
fprintf('���������vCR = %2.4f%%\n', CR*100)


%%   �e��j�S�x���G��
figure; % �}�ҷs��ø�ϪŶ�

plot(y(  1: 50,1),y(  1: 50,2),'ro',...
     y( 51:100,1),y( 51:100,2),'go',...
     y(101:150,1),y(101:150,2),'bo');   
     % �Hplotø�ϫ��O���O�e�Xclass1~3���Ĥ@�P�ĤG�S�x�C

title('Scatter Plot');                              % �ϦW��
legend('class1', 'class2', 'class3');               % ���O�и�����
xlabel('Feature3');                                 % �S�x�и�����
ylabel('Feature4');                                 % �S�x�и�����




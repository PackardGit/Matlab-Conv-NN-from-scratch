clear all;clc

%%
%wczytanie obrazków (20x20) oraz odpowiadaj¹cym im klas.
%wczytanie obrazów ucz¹cych
nr_test=10000;%liczba obrazów do destowania
[imgs_l labels_learn] = readMNIST('train-images.idx3-ubyte','train-labels.idx1-ubyte', nr_test*2, 0);
%wczytanie obrazów testowych
[imgs_t labels_test] = readMNIST('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte', nr_test, 0);
%%
%konwersja macierzy obrazów na wektory
for i=1:nr_test*2
imgs_learn(:,i)=reshape(imgs_l(:,:,i),[20*20,1]);
end

for i=1:nr_test
imgs_test(:,i)=reshape(imgs_t(:,:,i),[20*20,1]);
end
%%
%sieæ neuronowa
tic
%net = newff(imgs_learn,labels_learn,[40 10000],{'tansig' 'purelin' },'trainlm');
%newff([0 1, 0,1],[400 20 1],{'tansig' 'purelin' 'purelin'},'trainlm');% tworzenie sieci neuronow trainlm okresla algorytm traingd oznacza algrotm gradientowy, trainlm tez wykorzystuje gradient
net = feedforwardnet([400 100 10 10 10],'trainrp');
%net =cascadeforwardnet([80 40 10],'trainrp');
net.trainParam.epochs = 100; %krotnosc wykorzystywania zbioru uczacego
net.trainParam.showWindow = false;
net = train(net,imgs_learn, labels_learn');
learned = net(imgs_test);
%Y2 = sim(net,Xtest);
learned=learned';
blad=mse(learned,labels_test);

learned_round=round(learned);
roznica=labels_test-learned_round;
toc

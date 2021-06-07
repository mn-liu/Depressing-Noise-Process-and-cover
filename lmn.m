%首先对img2图片进行了resize，变成img3图片，再对图片进行机器学习简单去噪，对去噪的图像首先进行了增强，增强后再高斯滤波
%对img1图片进行像素点遍历，找出空白的地方，同时采用交互找到虚化的像素点位置，查看三通道阈值，设置阈值，使得找到虚化位置并保存
%对找到的位置进行像素点替换
%查看mse、rmse和psnr，查看最终图片效果
noiseAmp=2.1;%噪声的幅度
A = imread('img2.jpg');
B = imresize(A,[768,1024]);
imwrite(B,'img3.jpg');
sourceImage = imread('img3.jpg');
noiseFreqCut=0.42;%高频噪声的截止频率
filterStep=3;%巴特沃斯滤波器的截止频率
waveletLayerNum=5;%小波的分层系数
waveletType='db3';%小波基的种类
exampleImageAmount=10;%样本图片的个数
exampleImageNameFormat='exampleImage';%样本图片的名字格式
 
%设置样本阈值参考度的相关变量
t=linspace(0,pi,waveletLayerNum);%曲线采样点
exampleReferAmpi=1+cos(t);%样本阈值参考度
gainDropRate=50*ones(1,waveletLayerNum);%映射曲线在阈值处增益的下降速率
 
sourceImageDouble=im2double(sourceImage);%改变数据类型，便于对图像的操作
sourceNoise=sqrt(noiseAmp)*randn(size(sourceImageDouble));%加在图像上的噪声
[mLength,nLength,channel]=size(sourceImageDouble);%获取图像的尺寸
 
%噪声通过高通滤波器
%制作高通滤波器
%获取图像的中心距离矩阵
[xAxis,yAxis]=dftuv(mLength,nLength);
%高通滤波器的频域表达式
highPassFilterH=1./(1+(((noiseFreqCut*nLength)^2)./(xAxis.^2+yAxis.^2)).^filterStep);
 
highFreqNoise=[];%存储高频噪声的滤波结果
for tempVar=1:channel%对噪声的每个颜色层分别操作
    %取出一个单独的颜色层
    oneLayerSourceNoise=sourceNoise(:,:,tempVar);
    %单层图像通过高通滤波器
    simpleLayerFilResult=dftfilt(oneLayerSourceNoise,highPassFilterH);
    %卷积结果保存
    highFreqNoise=cat(3,highFreqNoise,simpleLayerFilResult);
end
 
noisyimage=imread('penguins.jpg');
noisyImageDouble=im2double(noisyimage);%改变数据类型，便于对图像的操作

noisyImage=sourceImageDouble;
%原始图像、噪声图像、加噪图像小波系数的比较
%原始图像的小波系数
[sourceImageWaveComp,sourceImageWaveSize]=wavedec2(sourceImageDouble,waveletLayerNum,waveletType);
%噪声图像小波系数
[noiseWaveComp,noiseWaveSize]=wavedec2(highFreqNoise,waveletLayerNum,waveletType);
%加噪图像的小波系数
[noisyImageWaveComp,noisyImageWaveSize]=wavedec2(noisyImage,waveletLayerNum,waveletType);
%直接归零的噪声处理***************
%拷贝一分加噪图像的系数矩阵
simpleDenoiseWaveComp=noisyImageWaveComp;
%确定系数向量近似分量的长度
lengthAppWaveCoef=noisyImageWaveSize(1,1)*noisyImageWaveSize(1,2)*noisyImageWaveSize(1,3);
%确定第一层细节分量的长度
lengthDet1WaveCoef=noisyImageWaveSize(2,1)*noisyImageWaveSize(2,2)*noisyImageWaveSize(2,3)*3;
%确定第二层细节分量的长度
lengthDet2WaveCoef=noisyImageWaveSize(3,1)*noisyImageWaveSize(3,2)*noisyImageWaveSize(3,3)*3;
%保留第一、二层细节分量，后面的直接置零
simpleDenoiseWaveComp(lengthAppWaveCoef+lengthDet1WaveCoef+lengthDet2WaveCoef:length(noisyImageWaveComp))=0;
%恢复处理过的噪声小波系数
simpleDenoiseResult=waverec2(simpleDenoiseWaveComp,noisyImageWaveSize,waveletType);
 
 
%简单机器学习思想的尝试*******************
 
%若干无噪声向量的和
exampleCompVector=zeros(1,waveletLayerNum);
for tempVar=1:exampleImageAmount
    %对应的图片名称
    tempImageName=cat(2,exampleImageNameFormat,' (',int2str(tempVar),').jpg');
    %无噪声系数向量的叠加
    exampleCompVector=exampleCompVector+waveCompDetEsti(tempImageName,waveletLayerNum,waveletType);
end
%求若干无噪声系数向量的平均值
exampleCompVector=exampleCompVector/exampleImageAmount;
%标记系数矩阵中细节分量的位置
tempCompLocal=noisyImageWaveSize(1,1)*noisyImageWaveSize(1,2)*noisyImageWaveSize(1,3);
%记录系数向量的副本
mLearningWaveletResult=noisyImageWaveComp;
%遍历每一层细节系数矩阵
for tempVar=1:waveletLayerNum
    %当前阶次细节分量向量的长度
    tempCompLength=noisyImageWaveSize(1+tempVar,1)*noisyImageWaveSize(1+tempVar,2)*noisyImageWaveSize(1+tempVar,3)*3;
    %对细节分量衰减
    mLearningWaveletResult(tempCompLocal:tempCompLocal+tempCompLength)=waveletDenoiseGain(noisyImageWaveComp(tempCompLocal:tempCompLocal+tempCompLength),exampleReferAmpi(tempVar)*exampleCompVector(tempVar),gainDropRate(tempVar));
    %更改系数矩阵位置
    tempCompLocal=tempCompLocal+tempCompLength;
end
%恢复到图像
mLearningResult=waverec2(mLearningWaveletResult,noisyImageWaveSize,waveletType);
%figure
%imshow(mLearningResult)
%title('简单机器学习算法去噪结果')
%imwrite(mLearningResult,'xiaobo.jpg')
%对图像进行增强
I = mLearningResult;
m = 16;
img = histeq(I,m);
%imshow(img,[]);
%title('均衡后的图像');
imwrite(img,'junheng.jpg')
img = imread('junheng.jpg');

% 对图像进行中值滤波
% 将图片分为R，G，B图片
R = img(:,:,1);
G = img(:,:,2);
B = img(:,:,3);
% 采用二维中值滤波函数medfilt2对图像滤波
R1 = gaosi(R);
G1 = gaosi(G);
B1 = gaosi(B);
% 合并RGB三通道
RGB(:,:,1)=R1(:,:,1);
RGB(:,:,2)=G1(:,:,1);
RGB(:,:,3)=B1(:,:,1);
%figure();
%subplot(1,2,1);
%imshow(img);
%title('原图像');
%subplot(1,2,2);
%imshow(RGB);
%title('高斯低通滤波');
imwrite(RGB,'gaosi1.jpg');
I = imread('img1.jpg');
R = imread('gaosi1.jpg');
[m,n,d]=size(I); 
%分别读取RGB
image_r = I(:,:,1);
image_g = I(:,:,2);
image_b = I(:,:,3);
for i=1:m 
    for j=1:n 
        if((I(i,j,1) == 255)&&(I(i,j,2) == 255)&&(I(i,j,3) == 255))
            I(i,j,1)=R(i,j,1);
            I(i,j,2)=R(i,j,2);
            I(i,j,3)=R(i,j,3);
        end
        if((j<310)&&(i<165))
            %if((I(i,j,1)+I(i,j,2)+I(i,j,3))>254)
            if(I(i,j,1)>190)
                I(i,j,1)=R(i,j,1);
                I(i,j,2)=R(i,j,2);
                I(i,j,3)=R(i,j,3); 
            end
        end
        if((j<941)&&(j>745)&&(i<266)&&(i>110))
            %if((I(i,j,1)+I(i,j,2)+I(i,j,3))>254)
            if(I(i,j,1)>200)
                I(i,j,1)=R(i,j,1);
                I(i,j,2)=R(i,j,2);
                I(i,j,3)=R(i,j,3); 
            end
        end
        
        if((j<370)&&(j>135)&&(i<585)&&(i>535))
            %if((I(i,j,1)+I(i,j,2)+I(i,j,3))>254)
            if(I(i,j,1)>200)
                I(i,j,1)=R(i,j,1);
                I(i,j,2)=R(i,j,2);
                I(i,j,3)=R(i,j,3); 
            end
        end
        if((j<475)&&(j>450)&&(i<186)&&(i>83))
            %if((I(i,j,1)+I(i,j,2)+I(i,j,3))>254)
            if(I(i,j,1)>200)
                I(i,j,1)=R(i,j,1);
                I(i,j,2)=R(i,j,2);
                I(i,j,3)=R(i,j,3); 
            end
        end
        if((702<j)&&(j<910)&&(615<i)&&(i<765))
             %if((I(i,j,1)+I(i,j,2)+I(i,j,3))>250)
             if((I(i,j,1)>50)&&(I(i,j,1)>50)&&I(i,j,1)>60)
                I(i,j,1)=R(i,j,1);
                I(i,j,2)=R(i,j,2);
                I(i,j,3)=R(i,j,3); 
            end
        end 
    end
end
imwrite(Z,'final.jpg')

X = imread('penguins.jpg');
X = im2double(X);
%X=rgb2gray(X);
Z = I;
Z = im2double(Z);
%Z=rgb2gray(Z);
figure;
imshow(Z); 
title('处理后图像');

X = double(X); 
Z = double(Z);
x = mse(X-Z);
MSE = (x(:,:,1) + x(:,:,1) + x(:,:,1))/3 ;
RMSE = sqrt(MSE);
PSNR = psnr(X,Z);
display(MSE);%均方根误差MSE
display(RMSE);%RMSE
display(PSNR);%峰值信噪比


function [img] = gaosi(image)
    d0=50;  %阈值
    [M ,N]=size(image);

    img_f = fft2(double(image));%傅里叶变换得到频谱
    img_f=fftshift(img_f);  %移到中间

    m_mid=floor(M/2);%中心点坐标
    n_mid=floor(N/2);  

    h = zeros(M,N);%高斯低通滤波器构造
    for i = 1:M
        for j = 1:N
            d = ((i-m_mid)^2+(j-n_mid)^2);
            h(i,j) = exp(-d/(2*(d0^2)));      
        end
    end

    img_lpf = h.*img_f;

    img_lpf=ifftshift(img_lpf);    %中心平移回原来状态
    img_lpf=uint8(real(ifft2(img_lpf)));  %反傅里叶变换,取实数部分
    
    img = img_lpf;
end
function [U, V] = dftuv(M, N)
% DFTUV Computes meshgrid frequency matrices.
% [U, V] = DFTUV(M, N) computes meshgrid frequency matrices U and V. U and
% V are useful for computing frequency-domain filter functions that can be
% used with DFTFILT. U and V are both M-by-N.
% more details to see the textbook Page 93
%
% [U，V] = DFTUV（M，N）计算网格频率矩阵U和V。 U和V对于计算可与DFTFILT一起使用的
% 频域滤波器函数很有用。 U和V都是M-by-N。更多细节见冈萨雷斯教材93页

% Set up range of variables.
% 设置变量范围
u = 0 : (M - 1);
v = 0 : (N - 1);

% Compute the indices for use in meshgrid.
% 计算网格的索引，即将网络的原点转移到左上角，因为FFT计算时变换的原点在左上角。
idx = find(u > M / 2);
u(idx) = u(idx) - M;
idy = find(v - N / 2);
v(idy) = v(idy) - N;

% Compute the meshgrid arrays.
% 计算网格矩阵
[V, U] = meshgrid(v, u);
end 
function g = dftfilt(f, H)
% DFTFILT performs frequency domain filtering.
%   G = DFTFILT(F, H) filters F in the frequency domain using the filter
%   transfer function H. The output, G, is the filtered image, which has
%   the same size as F. DFTFILT automatically pads F to be the same size as
%   H. Function PADEDESIZE can be used to determine an appropriate size
%   for H.
%   G = DFTFILT（F，H）使用滤波器传递函数H在频域中对输入图像F滤波。 输出G是滤波后的图像，
%   其大小与F相同。DFTFILT自动将F填充到与H相同的大小 ，PADEDESIZE函数可用于确定H的合适大小。
%
%   DFTFILT assumes that F is real and that H is a real, uncentered,
%   circularly-symmetric filter function.
%   DFTFILT假设F是实数，H是实数，未中心，循环对称的滤波函数。

% Obtain the FFT of the padded input.
% 获取填充之后的FFT变换
F = fft2(f, size(H, 1), size(H, 2));
% Perform filtering
% 滤波
g = real(ifft2(H .* F));
% Crop to orihinal size
% 剪切到原始尺寸
g = g(1 : size(f, 1), 1 : size(f, 2));
end
function [detMaxVector] = waveCompDetEsti(imageName,layerNum,waveletType)
%UNTITLED 返回一个向量，向量中依次是各个阶次细节分量的绝对值的最大值
%   图片名称的字符表示，小波分解的层数，小波基的种类
 
    %读取输入的图片
    imageMar=im2double(imread('img3.jpg'));
    %对输入的图片进行小波系数分解
    [waveComp,waveSize]=wavedec2(imageMar,layerNum,waveletType);
    %分别求小波系数记录最大值
    detMaxVector=[];
    for tempVar=1:layerNum
        detVector=[detcoef2('h',waveComp,waveSize,tempVar),detcoef2('v',waveComp,waveSize,tempVar),detcoef2('d',waveComp,waveSize,tempVar)];
        detMaxVector=cat(2,max(max(max(abs(detVector)))),detMaxVector);
    end
end 
function [result] = waveletDenoiseGain(operateVar,dropLocal,dropRate)
%UNTITLED 用于小波图像去噪的细节分量系数映射
%   用反正切函数映射
%保留符号
result=-atan(dropRate*(abs(operateVar)-dropLocal))/pi+0.5;
result=operateVar.*result;
end

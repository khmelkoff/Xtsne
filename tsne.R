###############################################################################
# tSNE experiments
# author: khmelkoff
# refs: http://lvdmaaten.github.io/tsne/code/tSNE_matlab.zip
#       https://cran.r-project.org/src/contrib/tsne_0.1-2.tar.gz
###############################################################################

library(fields)
library(ggplot2)
### algorithm and MNIST experiment ############################################

### Load Data
source("loadMNIST.R")
load_mnist()

X <- train$x
set.seed(123)
index <- sample(1:60000, 6000, replace=FALSE)
X <- X[index,]
labels <- train$y
labels <- labels[index]
X <- X/255


### Display digits
source("display.R")
images<-t(X)
display(images[,1:20]);

### PCA
pca <- prcomp(X)
X <- pca$x[,1:30]

### Inintialize
n = nrow(X)
k = 2 # low dimension
Y = matrix(0.0001*rnorm(n * k),n)

### Distance calc
D <- (rdist(X))^2
D <- as.matrix(D)

### Perplexity and Kernel calc
perpCalc <- function(D, gamma) {
    P <- exp(-D * gamma) # kernel calc    
    sumP <- sum(P)
    PP <- log(sumP) + gamma * sum(D * P) / sumP
    r = {}
    r$PP = PP
    r$P = P/sumP
    r
}

### X distribution
distrX <- function(D, perplexity, tresh = 1e-5){
        
    P = matrix(0, n, n )    	
    gamma = rep(0.5, n)
    logPerp = log(perplexity)
    #print(logPerp)
        
    for (i in 1:n){
        gammamin = -Inf
        gammamax = Inf
        Di = D[i, -i]
        
        pc <- perpCalc(Di, gamma[i])
        PP = pc$PP; 
        PR = pc$P
        perpDiff = PP - logPerp;
                
        tries = 0;
            
        while(abs(perpDiff) > tresh && tries < 50){
            if (perpDiff > 0){
                gammamin = gamma[i]
                if (is.infinite(gammamax)) gamma[i] = gamma[i] * 2
                else gamma[i] = (gamma[i] + gammamax)/2
            } else{
                gammamax = gamma[i]
                if (is.infinite(gammamin))  gamma[i] = gamma[i]/ 2
                else gamma[i] = ( gamma[i] + gammamin) / 2
            }
                
            pc <- perpCalc(Di, gamma[i])
            PP <- pc$PP
            PR <- pc$P
            perpDiff <- PP - logPerp;
            tries <- tries + 1
        }	
        P[i,-i]  = PR	
    }	
        
    r = {}
    r$P = P
    r$gamma = gamma
    sigma = sqrt(1/gamma)
    
    message('sigma summary: ', 
            paste(names(summary(sigma)),':',
                  summary(sigma),'|',collapse=''))
        
    r 
}

### Neighbors embedding
tsne <- function(D, perplexity, max_iter = 1000, min_cost=0, epoch_callback=NULL, epoch=100 ){
    
        momentum = .5               # initial momentum
        final_momentum = .8         # value to which momentum is changed
        mom_switch_iter = 250       # iteration at which momentum is changed
        
        epsilon = 500               # initial learning rate
        min_gain = .01              # minimum gain for delta-bar-delta
        initial_P_gain = 4          # early exaggeration multiplier
    
        eps = 2^(-52)               # typical machine precision
        
        # Heigh dim joit probability calc
        P = distrX(D, perplexity, 1e-5)$P
        P = .5 * (P + t(P))         # symmetrize P
        
        P[P < eps]<-eps
        P = P/sum(P) 
        
        P = P * initial_P_gain
        grads =  matrix(0,nrow(Y),ncol(Y))
        incs =  matrix(0,nrow(Y),ncol(Y))
        gains = matrix(1,nrow(Y),ncol(Y))
        
        
        for (iter in 1:max_iter){
            
            if (iter %% epoch == 0) { # epoch
                cost =  sum(apply(P * log(P/Q),1,sum))
                message("Epoch: Iteration #",iter," error is: ",cost)
                if (cost < min_cost) break
                if (!is.null(epoch_callback)) epoch_callback(Y)
            }
            
            # Compute joint probability that point i and j are neighbors
            # Student-t distribution
            num = 1/(1 + (rdist(Y))^2)
            sum_ydata = apply(Y^2, 1, sum)
            # MatLab variant:
            # num =  1/(1 + sum_ydata +    sweep(-2 * Y %*% t(Y),2, -t(sum_ydata))) 
            
            diag(num)=0 # Set diagonal to zero
            Q = num / sum(num) # Normalize to get probabilities
            Q[Q < eps] = eps
            
            # Compute the gradients (faster implementation)
            L = (P - Q) * num;
            grads = 4 * (diag(apply(L, 1, sum)) - L) %*% Y
                    
            
            # Update the solution
            gains = (gains + .2) * abs(sign(grads) != sign(incs)) 
            + gains * .8 * abs(sign(grads) == sign(incs))		
            gains[gains < min_gain] = min_gain
            
            incs = momentum * incs - epsilon * (gains * grads)
            
            Y = Y + incs
            Y = sweep(Y,2,apply(Y,2,mean))
            
            # Update the momentum if necessary
            if (iter == mom_switch_iter) momentum = final_momentum
            
            if (iter == 100) P = P/4
    }
    Y
}


tstart <- Sys.time()
yy_6000 <- tsne(D,40)
tend <- Sys.time()
tend-tstart
#save(yy_6000, file="Data/yy_6000.RData")

### ggplot visualization
dat <- data.frame(Dim1=yy_6000[,1], Dim2=yy_6000[,2])
ggplot(dat, aes(x=Dim1, y=Dim2, color=labels)) +
    geom_point(aes(colour = factor(labels)), size=2) 
    

### 900 words distributed representation ######################################
data <- read.csv("Data/samplewordembedding.csv", as.is=TRUE)
labels <- row.names(data)
X <- data

pca <- prcomp(X)
X <- pca$x[,1:30]

### Inintialize
n = nrow(X)
k = 3 # low dimension
Y = matrix(0.0001*rnorm(n * k),n)

### Distance calc
D <- (rdist(X))^2
D <- as.matrix(D)

y_word_1000 <- tsne(D,40)
save(y_word_1000, file="Data/y_word_1000.RData")

dat <- data.frame(Dim1=y_word_1000[,1], Dim2=y_word_1000[,2],Dim3=y_word_1000[,3])
pic <- ggplot(dat, aes(x=Dim1, y=Dim2, color=Dim3, label=labels)) 
pic <- pic + geom_text(size=4)
pic <- pic + theme_bw()
pic <- pic + scale_color_gradient2(low="blue", mid="green", high="red")
pic

### k-menas clustering ########################################################
cl <- kmeans(X,90)
clusters <- cl$cluster

xcl <- cbind(clusters,X)
cln <- c(1,2,4,7,14,15,19,20,26,27,28,31,33,34,55) # Select your clusters!
newX <- xcl[clusters %in% cln,]
newX <- newX[,-1]
labels <- row.names(newX)

### Inintialize
n = nrow(newX)
k = 3 # low dimension
Y = matrix(0.0001*rnorm(n * k),n)

### Distance calc
D <- (rdist(newX))^2
D <- as.matrix(D)

y_word_147 <- tsne(D,40)
save(y_word_147, file="Data/y_word_147.RData")

dat <- data.frame(Dim1=y_word_147[,1], Dim2=y_word_147[,2],Dim3=y_word_147[,3])
pic <- ggplot(dat, aes(x=Dim1, y=Dim2, color=Dim3, label=labels)) 
pic <- pic + geom_text(size=4)
pic <- pic + theme_bw()
pic <- pic + scale_color_gradient2(low="blue", mid="green", high="red")
pic

### tsne library experiment ###################################################
library(tsne)
yd <- tsne(X, perplexity=40)

library(ggplot2)
dat <- data.frame(Dim1=yd[,1], Dim2=yd[,2])
ggplot(dat, aes(x=Dim1, y=Dim2, color=labels)) +
    geom_point(aes(colour = factor(labels)), size=3) 

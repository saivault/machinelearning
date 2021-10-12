#### MATHEMATICAL FORMULATION OF REGRESSION
Dataset D where X is the input matrix consisting of n data points with m features each and Y is the output matrix <br><br>
$$
   D = \langle x_i,y_i \rangle_{i=1}^n \hspace{1 cm} \text{where} \hspace{1 mm} x_i \in \mathbb{R}^{n \times m} \hspace{2 mm} \text{and} \hspace{1 mm} y_i \in \mathbb{R} \hspace{10 cm}
$$

# Linear Regression

### GOAL OF LINEAR REGRESSION
Find a line (or) hyperplane that best fits the given data. <br>
$ \hspace{1 cm} \hat{y} = f(x) = w_0x_0 +  w_1x_1 +  w_2x_2 +  w_3x_3 + .......  w_mx_m $ <br>
Finding the weights $ w_0, w_1, w_2, ...... w_m $ is our ultimate goal <br>

<b>What do you mean by best fit ?</b><br>
$ Error_i = y_i - \hat{y_i} $ <br>
Minimise the $ \sum_{i}Error_i $ across our training data.<br> Since the errors can be negative, lets try to minimise $ \sum_{i}Error_i^2 $ 

### MATHEMATICAL FORMULATION
Find optimal $ w^* $ such that it minimises our errors <br>
$$
    w^* = \text{arg} \min_{w}  \sum_{i}(y_i - \hat{y_i})^2 = \text{arg} \min_{w} \Big[ \sum_{i=1}^{n}(y_i - w^Tx_i)^2 \Big]
$$
=> Because of this squared loss function, Linear regression is often referred to as <b>Ordinary Least Squares</b> (OLS) or Linear Least Squares. (LLS)

argmin is argument of the minimum. The simplest example is given by <br>
$ \mathrm{arg} \min_{𝑥} f(x) $ is the value of $x$ for which $f(x)$ attains its minimum.

#### REGULARISATION
$$
    w^* = \text{arg} \min_{w} \Big[ \sum_{i=1}^{n}(y_i - w^Tx_i)^2 \Big] + \lambda \| w \|_2^2 \hspace{1 cm} \text{if we use } L_2 \text{ regulariser}
$$

Linear regression with $L_1$ regularization is also referred to as <i><b>Lasso Regression</b></i> <br>
Linear regression with $L_2$ regularization is also referred to as <i><b>Ridge Regression</b></i> <br>
Linear regression with $L_1 + L_2$ regularization is also referred to as <i><b>ElasticNet Regression</b></i>.

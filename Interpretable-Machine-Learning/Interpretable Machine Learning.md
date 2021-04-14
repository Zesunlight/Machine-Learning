# Interpretable Machine Learning 可解释的机器学习

## Introduction

### Terminology

- Machine Learning is a set of methods that allow computers to learn from data to make and improve predictions (for example cancer, weekly sales, credit default). Machine learning is a paradigm shift from “normal programming” where all instructions must be explicitly given to the computer to “indirect programming” that takes place through providing data.

  机器学习(Machine Learning) 是⼀套⽅法，能够允许计算机从数据中学习，以做出和改进预测(例如癌症、每周销售、信⽤违约)。机器学习是从“常规编程” (Normal Programming) 到“间接编程”(Indirect Programming) 的⼀种范式转换，“常规编程” 是指所有指令都必须显式地提供给计算机，⽽“间接编程” 是通过提供数据来实现的。

## Interpretability 可解释性

- Interpretability is the degree to which a human can understand the cause of a decision.

  可解释性是⼈们能够理解决策原因的程度。

- Interpretability is the degree to which a human can consistently predict the model’s result.

  可解释性是指⼈们能够⼀致地预测模型结果的程度。

### Taxonomy of Interpretability Methods 可解释性方法的分类

- Intrinsic 本质的 or Post-hoc 事后的

  Intrinsic interpretability refers to machine learning models that are considered interpretable due to their simple structure, such as short decision trees or sparse linear models. Post hoc interpretability refers to the application of interpretation methods after model training. Permutation feature importance is, for example, a post hoc interpretation method.

  本质的可解释性是指由于结构简单⽽被认为是可解释的机器学习模型，如短的决策树或稀疏线性模型；事后解释性是指模型训练后运⽤解释⽅法，例如，置换特征重要性是⼀种事后解释⽅法。

### Scope of Interpretability 可解释性范围

#### Algorithm Transparency 算法透明度

Algorithm transparency is about how the algorithm learns a model from the data and what kind of relationships it can learn. If you use convolutional neural networks to classify images, you can explain that the algorithm learns edge detectors and filters on the lowest layers. This is an understanding of how the algorithm works, but not for the specific model that is learned in the end, and not for how individual predictions are made. Algorithm transparency only requires knowledge of the algorithm and not of the data or learned model. This book focuses on model interpretability and not algorithm transparency. Algorithms such as the least squares method for linear models are well studied and understood. They are characterized by a high transparency. Deep learning approaches (pushing a gradient through a network with millions of weights) are less well understood and the inner workings are the focus of ongoing research. They are considered less transparent.

算法透明度是指算法如何从数据中学习模型，以及它可以学习到什么样的关系。如果使⽤卷积神经⽹络对图像进⾏分类，则可以解释该算法在最底层学习边缘检测器和滤波器。这是对算法如何⼯作的理解，但既不是对最终学习的特定模型的理解，也不是对如何做出单个预测的理解。算法的透明度只需要对算法的了解，⽽不需要对数据或学习模型的了解。这本书的重点是模型的可解释性，⽽不是算法的透明度。线性模型的最⼩⼆乘法等算法已被深⼊地研究和理解，它们的特点是透明度⾼。深度学习⽅法(通过具有数百万权重的⽹络推动梯度) 不太容易理解，对其内部⼯作机制的探究是正在进⾏的研究重点，它们被认为是低透明度的。

#### Human-friendly Explanations

- Explanations are contrastive 对比性

  Humans do not want a complete explanation for a prediction, but want to compare what the differences were to another instance’s prediction (can be an artificial one).
  
  ⼈类不希望对预测有⼀个完整的解释，⽽是希望将不同之处与另⼀个实例(可以是⼈⼯的) 的预测进⾏⽐较。
  
- Explanations are selected 选择几个原因

  People do not expect explanations that cover the actual and complete list of causes of an event. We are used to selecting one or two causes from a variety of possible causes as THE explanation.

  ⼈们不希望对涵盖事件的实际原因和完整原因进⾏解释。我们习惯于从各种可能的原因中选择⼀个或两个原因作为解释。

  What it means for interpretable machine learning: Make the explanation very short, give only 1 to 3 reasons, even if the world is more complex.

  它对可解释机器学习意味着什么：解释要简短，即使真实情况很复杂，但只给出1 到3 个原因。

- For machine learning models, it is advantageous if a good prediction can be made from different features. Ensemble methods that combine multiple models with different features (different explanations) usually perform well because averaging over those “stories” makes the predictions more robust and accurate. But it also means that there is more than one selective explanation why a certain prediction was made.

  对于机器学习模型，如果能根据不同的特征做出⼀个好的预测是有利的。将使⽤不同的特征(不同的解释) 的多个模型结合在⼀起的集成⽅法通常表现良好，进⾏平均可以使预测更加鲁棒和准确。但这也意味着有不⽌⼀个选择性的解释——为什么做出了某种预测。
  
- Explanations focus on the abnormal. 解释的重点是异常


## Interpretable Models

### Linear Regression

- $$
  y=\beta_{0}+\beta_{1} x_{1}+\ldots+\beta_{p} x_{p}+\epsilon \\
  \hat{\beta}=\arg \min _{\beta_{0}, \ldots, \beta_{p}} \sum_{i=1}^{n}\left(y^{(i)}-\left(\beta_{0}+\sum_{j=1}^{p} \beta_{j} x_{j}^{(i)}\right)\right)^{2}
  $$

- The linear regression model forces the prediction to be a linear combination of features, which is both its greatest strength and its greatest limitation. Linearity leads to interpretable models. Linear effects are easy to quantify and describe. They are additive, so it is easy to separate the effects.

  线性回归模型使预测成为特征的线性组合，这既是其最⼤的优势，也是其最⼤的局限。线性导致其为可解释模型。线性效应易于量化和描述，它们是可加的，因此很容易分离效应。

- When the features have been standardised (mean of zero, standard deviation of one), the intercept reflects the predicted outcome of an instance where all features are at their mean value.

  当特征标准化(均值为0，标准差为1) 时，截距就将反映当所有特征都处于其均值时的实例的预测结果。
  
- All the interpretations always come with the footnote that “all other features remain the same”. This is because of the nature of linear regression models. The predicted target is a linear combination of the weighted features. The estimated linear equation is a hyperplane in the feature/target space (a simple line in the case of a single feature). The weights specify the slope (gradient) of the hyperplane in each direction. The good side is that the additivity isolates the interpretation of an individual feature effect from all other features. That is possible because all the feature effects (= weight times feature value) in the equation are combined with a plus. On the bad side of things, the interpretation ignores the joint distribution of the features. Increasing one feature, but not changing another, can lead to unrealistic or at least unlikely data points.

  所有的解释总是伴随“所有其他特征保持不变”，这是因为线性回归模型的性质。预测⽬标是加权特征的线性组合。估计的线性⽅程是特征/⽬标空间中的超平⾯(在单个特征的情况下为线)。权重指定每个⽅向上超平⾯的斜率(梯度)。好的⼀⽅⾯是，可加性将单个特征的解释与所有其他特征隔离开来。这是可能的，因为⽅程式中的所有特征效应(= 权重乘以特征值) 都是⽤加号组合在⼀起。坏的⼀⾯是，这种解释忽略了特征的联合分布。增加⼀个特征，但不改变另⼀个特征，这可能不合实际或者是不太可能的数据点。

#### Sparse Linear Models 稀疏线性模型

You might not have just a handful of features, but hundreds or thousands. And your linear regression models? Interpretability goes downhill. You might even find yourself in a situation where there are more features than instances, and you cannot fit a standard linear model at all. The good news is that there are ways to introduce sparsity (= few features) into linear models.

在真实的情况下你可能不只是拥有少数个特征，⽽是拥有成百上千个特征。这种情况下你的线性回归模型的解释能⼒就会下降。你甚⾄可能会发现处于这样⼀种情况，即特征⽐实例多，并且根本⽆法拟合标准线性模型。幸运的是，有很多⽅法可以将稀疏性(即很少的特征) 引⼊线性模型。

##### Lasso

- L1正则化
  $$
  \min _{\boldsymbol{\beta}}\left(\frac{1}{n} \sum_{i=1}^{n}\left(y^{(i)}-x_{i}^{T} \boldsymbol{\beta}\right)^{2}+\lambda\|\boldsymbol{\beta}\|_{1}\right)
  $$

#### Other Methods for Sparsity in Linear Models 线性模型中稀疏性的其他方法

##### Pre-processing methods 预处理

- Manually selected features ⼿动选择特征

  You can always use expert knowledge to select or discard some features. The big drawback is that it cannot be automated and need be an expert.

  你可以始终使⽤专家知识来选择或放弃某些特征。最⼤的缺点是它不能被⾃动化，需要与理解数据的⼈员取得联系。

- Univariate selection 单变量选择

  An example is the correlation coefficient. You only consider features that exceed a certain threshold of correlation between the feature and the target. The disadvantage is that it only considers the features individually. Some features might not show a correlation until the linear model has accounted for some other features. Those ones you will miss with univariate selection methods.

  相关系数就是⼀个例⼦。你只考虑超过特征和⽬标之间相关性阈值的特征。缺点是它只是单独地考虑单个特征。在线性模型考虑了其他⼀些特征之前，某些特征可能不会显⽰相关性。⽽使⽤单变量选择⽅法你会错过这些特征。

##### Step-wise methods

- Forward selection

  Fit the linear model with one feature. Do this with each feature. Select the model that works best (e.g. highest R-squared). Now again, for the remaining features, fit different versions of your model by adding each feature to your current best model. Select the one that performs best. Continue until some criterion is reached, such as the maximum number of features in the model.

  ⽤⼀个特征拟合线性模型，对每个特征都执⾏此操作。选择最有效的模型(例如，最⾼R-平⽅)；然后，对于其余的特征，通过将每个特征添加到当前的最佳模型中来拟合模型的不同版本，选择表现最好的⼀个；重复操作，直到达到某个条件，例如最⼤特征数。

- Backward selection

  Similar to forward selection. But instead of adding features, start with the model that contains all features and try out which feature you have to remove to get the highest performance increase. Repeat this until some stopping criterion is reached.

  类似于向前选择。但是，不是添加特征，⽽是从包含所有特征的模型开始，尝试删除某个特征以达到性能最⼤程度的提⾼；重复此操作，直到达到某个停⽌标准。

### Logistic Regression 逻辑回归

### GLM, GAM and more

#### Non-Gaussian Outcomes -GLMs 非高斯结果输出

- ⼴义线性模型(Generalized Linear Models，GLMs)

- Keep the weighted sum of the features, but allow non-Gaussian outcome distributions and connect the expected mean of this distribution and the weighted sum through a possibly nonlinear function. For example, the logistic regression model assumes a Bernoulli distribution for the outcome and links the expected mean and the weighted sum using the logistic function.

  保留特征的加权和，但允许⾮⾼斯结果分布，并通过可能的⾮线性函数连接该分布的期望均值与加权和。例如，逻辑回归模型假设结果为伯努利分布，并使⽤Logit 函数连接期望均值与加权和。

- $$g\left(E_{Y}(y | x)\right)=\beta_{0}+\beta_{1} x_{1}+\ldots \beta_{p} x_{p}$$

  The link function g, the weighted sum XT  (sometimes called linear predictor) and a probability distribution from the exponential family that defines EY .

  GLM 由三个部分组成：连接函数$g$、加权和$X^T\beta$  (有时称为线性预测因⼦) 以及定义$E_Y$ 指数族的概率分布。

#### Interactions 交互

- Before you fit the linear model, add a column to the feature matrix that represents the interaction between the features and fit the model as usual.

  在拟合线性模型之前，要在特征矩阵中添加⼀列，表⽰特征之间的交互，并像往常⼀样拟合模型。

#### Nonlinear Effects - GAMs 非线性效应

- ⼴义加性模型(Generalized Additive Models，GAMs)
-  $$g\left(E_{Y}(y | x)\right)=\beta_{0}+f_{1}\left(x_{1}\right)+f_{2}\left(x_{2}\right)+\ldots+f_{p}\left(x_{p}\right)$$

### Decision Tree 决策树

- CART

### Decision Rules 决策规则

- A decision rule is a simple IF-THEN statement consisting of a condition (also called antecedent) and a prediction.

  决策规则(Decision Rules) 是⼀个简单的IF-THEN 语句，由条件(也称为先⾏条件) 和预测组成。

- Support or coverage of a rule 支持度 / 覆盖率

  The percentage of instances to which the condition of a rule applies is called the support.

  规则条件适⽤的实例所占的百分⽐称为⽀持度。

- Accuracy or confidence of a rule 准确性 / 置信度

  The accuracy of a rule is a measure of how accurate the rule is in predicting the correct class for the instances to which the condition of the rule applies.

  规则的准确性是衡量规则在规则条件适⽤的实例预测正确类别时的准确性的指标。

- Usually there is a trade-off between accuracy and support: By adding more features to the condition, we can achieve higher accuracy, but lose support.

  通常在准确性和⽀持度之间有⼀个权衡：通过在条件中添加更多的特征，我们可以获得更⾼的准确性，但会失去⽀持度。

- 从数据中学习规则

  1. OneR 从单个特征中学习规则。OneR 的特点在于其简单性、可解释性、并且可以⽤作基准。
  2. 顺序覆盖(Sequential covering) 是⼀种通⽤过程，可以迭代地学习规则并删除新规则覆盖的数据点。许多规则学习算法都使⽤此过程。
  3. 贝叶斯规则列表(Bayesian Rule List) 使⽤贝叶斯统计将预先挖掘的频繁模式组合到决策列表中。使⽤预先挖掘模式是许多规则学习算法所使⽤的常见⽅法。

### RuleFit



## Reference 参考

- https://christophm.github.io/interpretable-ml-book/
- https://github.com/MingchaoZhu/InterpretableMLBook/tree/35aec6a6b27125d4961a1a1aa82b675cbbc35618
- https://scikit-learn.org/stable/modules/tree.html
- https://github.com/oracle/Skater
- 
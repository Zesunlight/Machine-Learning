> [机器学习基石上](https://www.coursera.org/learn/ntumlone-mathematicalfoundations) (Machine Learning Foundations)---Mathematical Foundations
> [Hsuan-Tien Lin, 林轩田](https://www.coursera.org/instructor/htlin)，副教授 (Associate Professor)，资讯工程学系 (Computer Science and Information Engineering)

## Feasibility of Learning

### Learning is Impossible?

- Two Controversial Answers  多种合理的方式得到不同的答案
- no-free-lunch problems


### Probability to the Rescue

- Inferring Something Unknown → sample
- 样本(in-sample)的概率与整体(out-of-sample)的概率大概是接近的
- Hoeffding's Inequality 
  - ![Hoeffding's Inequality](Hoeffding's Inequality.png)
  - ![Hoeffding's Inequality 2](Hoeffding's Inequality 2.png)

### Connection to Learning

- 抓弹珠类比学习

  ![connection to learning](connection to learning.png)

  - 抓的一把弹珠是已知数据
  - 橙色的代表错误
  - 抽样测试，测试集上的正确率

- ![added components](added components.png)

- 公式表述

  ![the formal guarantee](the formal guarantee.png)

- Verification

  - Verification of One h 

    ![verification](verification.png)

  - The Verification Flow 

    ![verification flow](verification flow.png)

### Connection to Real Learning

- BAD sample: $E_{in}$ and $E_{out}$ far away (can get worse when involving choice)

- BAD Data for One h: $E_{in}(h)$ and $E_{out}(h)$ far away

  - 不好的几率很小
  - ![bad data](bad data.png)

- BAD Data for Many h 

  - ![bad data for many h](bad data for many h.png)

  - for $M$ hypotheses, bound of $\mathbb P_{\mathcal D}[{\color{orange}{BAD}} \;\mathcal D]$ 

    ![bound of bad data](bound of bad data.png)

- The Statistical Learning Flow

  ![statistical learning flow](statistical learning flow.png)




learning possible if $|\mathcal H|$ finite and $E_{in}(g)$ small








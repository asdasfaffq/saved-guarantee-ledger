# Proof Package — SAVED anytime-valid recall guarantees

诚实声明：本文件区分"严格 close"与"仅在加强假设下成立/启发式"。核心发现（命题3）：当前**实现**采用的"聚合加权平均时龄 inflation"在一般情况下**不是严格有效**的；严格有效的构造是**逐样本平移**（per-sample shift），它在有界漂移下严格 close。下面逐条给出。

## 通用记号与设定
- 过滤 $\mathcal F_i=\sigma(X_1,\dots,X_i)$，$\mathcal F_0=\{\emptyset,\Omega\}$。
- 召回估计量来自洞察 $\mathrm{recall}(\tau)=\Pr(s(X)\ge\tau\mid g(X)=1)=\mathbb E[\mathbf 1\{s(X)\ge\tau\}\mid g(X)=1]$：对 oracle 标注为正例的样本，其"是否被选中"指示量 $X_i=\mathbf 1\{s\ge\tau\}\in\{0,1\}$ 的条件均值即为相应快照召回。未知的"语料正例总数"分母被消去（条件均值无需该计数）。
- 下注系数 $\lambda_i$ **可预测**（$\mathcal F_{i-1}$-可测），取值 $[0,\Lambda]$，$\Lambda\le 1$（实现用 $\Lambda=0.6$）；含 empirical-Bernstein 插值 $\lambda_i=\mathrm{clip}\!\big(\sqrt{2\ln(1/\delta)/(\hat v_{i-1}(i)} ),0,\Lambda\big)$，仅用 $X_1..X_{i-1}$ 故可预测。
- 权重 $w_i\in[0,1]$ 可预测。

---

## 命题1（无漂移：anytime-valid 召回下界）
**Status: PROVABLE AS STATED.**

**陈述.** 设 $X_1,X_2,\dots$ i.i.d. $\mathrm{Bernoulli}(\mu)$。对 $m\in[0,1]$ 定义
$$K_t(m)=\prod_{i=1}^{t}\big(1+\lambda_i(X_i-m)\big),\quad K_0(m)=1.$$
则下界 $L_t=\sup\{m\in[0,1]:K_t(m)\ge 1/\delta\}$（约定 $\sup\emptyset=0$）满足 $\Pr(\exists t:\ \mu<L_t)\le\delta$。

**证明.**
1. *非负性.* 对 $X_i\in\{0,1\},m\in[0,1],\lambda_i\in[0,1]$：若 $X_i=0$，因子 $=1-\lambda_i m\ge 1-\lambda_i\ge0$；若 $X_i=1$，因子 $=1+\lambda_i(1-m)\ge1>0$。故 $K_t(m)\ge0$。
2. *上鞅（在 $H_0:\mu\le m$）.* $\lambda_t$ 可预测，
$$\mathbb E[K_t(m)\mid\mathcal F_{t-1}]=K_{t-1}(m)\big(1+\lambda_t(\mu-m)\big)\le K_{t-1}(m),$$
因 $\lambda_t\ge0,\ \mu-m\le0$。又 $K_0=1$。故为非负上鞅，$\mathbb E K_0=1$。
3. *$m\mapsto K_t(m)$ 非增.* 每个因子 $1+\lambda_i(X_i-m)$ 关于 $m$ 非增且非负；非负非增函数之积非增。故 $\{m:K_t(m)\ge1/\delta\}$ 是形如 $[0,L_t]$ 的（闭）区间，且 $\mu<L_t\Rightarrow K_t(\mu)\ge1/\delta$。
4. *Ville.* 取真值 $m=\mu$（此时 $H_0$ 以等号成立），$K_t(\mu)$ 是非负上鞅，$\mathbb E K_0=1$，由 Ville 不等式 $\Pr(\exists t:K_t(\mu)\ge1/\delta)\le\delta$。
5. 结合步骤3：$\{\exists t:\mu<L_t\}\subseteq\{\exists t:K_t(\mu)\ge1/\delta\}$，故 $\Pr(\exists t:\mu<L_t)\le\delta$。∎

---

## 引理2（加权下注保持有效性）
**Status: PROVABLE AS STATED.**

**陈述.** 以 $1+w_i\lambda_i(X_i-m)$ 替换因子，$w_i\in[0,1]$ 可预测。则结论同命题1。

**证明.** 令有效系数 $\lambda_i'=w_i\lambda_i\in[0,\Lambda]\subseteq[0,1]$，仍可预测。非负性与上鞅性同命题1（$X_i=0$ 时因子 $=1-\lambda_i' m\ge1-\lambda_i'\ge0$；$\mathbb E[\cdot\mid\mathcal F_{t-1}]=1+\lambda_t'(\mu-m)\le1$）。其余照搬命题1。∎

---

## 命题3（漂移下的有效召回下界）—— 核心
样本 $i$ 取自更早快照 $t_i\le s$（$s:=t_{\text{now}}$ 为目标快照时间），$\mu_i:=\mathbb E[X_i\mid\mathcal F_{i-1}]=\mathrm{recall}_{t_i}$。目标量 $\mu_s:=\mathrm{recall}_{s}$。
**有界漂移假设 (A-drift).** 存在已知 $\kappa\ge0$ 使 $|\mu_i-\mu_s|\le b_i:=\kappa\,(s-t_i)$（高斯模型下 $\kappa\le\varepsilon\,\phi_{\max}$，$\phi_{\max}=0.3989$）。

### 3a 逐样本平移构造 — **Status: PROVABLE AS STATED**
**构造.** 令 $\tilde b_i=\min(b_i,1)$，
$$\tilde K_t(m)=\prod_{i=1}^{t}\Big(1+w_i\lambda_i\big(X_i-\underbrace{\min(m+b_i,1)}_{=:m_i}\big)\Big),\qquad
\tilde L_t=\sup\{m:\tilde K_t(m)\ge1/\delta\}.$$
**陈述.** 在 (A-drift) 下 $\Pr(\exists t:\ \mu_s<\tilde L_t)\le\delta$。

**证明.**
1. *非负性.* $m_i\in[0,1]$，$w_i\lambda_i\in[0,1]$：$X_i=0$ 时因子 $=1-w_i\lambda_i m_i\ge1-w_i\lambda_i\ge0$；$X_i=1$ 时 $\ge1>0$。
2. *上鞅（在 $H_0^{\text{now}}:\mu_s\le m$）.* 由 (A-drift) 的**上侧**：$\mu_i\le\mu_s+b_i$。若 $m+b_i\le1$ 则 $m_i=m+b_i$，
$$\mathbb E[\,\cdot\mid\mathcal F_{i-1}]=1+w_i\lambda_i(\mu_i-m_i)=1+w_i\lambda_i\big(\mu_i-m-b_i\big)\le 1+w_i\lambda_i\big((\mu_s+b_i)-m-b_i\big)=1+w_i\lambda_i(\mu_s-m)\le1,$$
末步用 $H_0^{\text{now}}$。若 $m+b_i>1$ 则 $m_i=1\ge\mu_i$，故 $\mu_i-m_i\le0$，因子条件期望 $\le1$ 仍成立。两情形皆为上鞅增量。
3. *$m\mapsto\tilde K_t(m)$ 非增*（每因子关于 $m$ 非增且非负），故水平集为 $[0,\tilde L_t]$，且 $\mu_s<\tilde L_t\Rightarrow\tilde K_t(\mu_s)\ge1/\delta$。
4. *Ville.* 取 $m=\mu_s$：由步骤2，$\tilde K_t(\mu_s)$ 是非负上鞅（$\mathbb E K_0=1$），$\Pr(\exists t:\tilde K_t(\mu_s)\ge1/\delta)\le\delta$。结合步骤3得证。∎

*注.* 仅用到 (A-drift) 的**上侧** $\mu_i\le\mu_s+b_i$（对下界有效性正确的方向），对单调/双向漂移均成立；无需下侧。$w_i$ 任意可预测（含 $w_i=\gamma^{s-t_i}$）只影响功效不影响有效性。

### 3b 聚合"加权平均时龄"inflation（**当前实现**）— **Status: NOT JUSTIFIED AS IMPLEMENTED**
**实现构造.** 用未平移的 $K_t(m)=\prod(1+w_i\lambda_i(X_i-m))$ 之 $L_t=\sup\{m:K_t(m)\ge1/\delta\}$，再减
$$\Delta_t^{\text{mean}}=\kappa\cdot\frac{\sum_i w_i\,(s-t_i)}{\sum_i w_i}\quad(\text{加权平均时龄}),$$
报告 $L_t-\Delta_t^{\text{mean}}$。

**反例性分析（为何不严格）.** 未平移的 $K_t(m)$ 仅在"$\mu_i\le m\ \forall i$"时为上鞅；故 $L_t$ 是确定性量 $M^\*:=\sup_i\mu_i$ 的 anytime-valid **下界**：$\Pr(\exists t:M^\*<L_t)\le\delta$，即高概率 $L_t\le M^\*$。在召回随时间下降的漂移下 $M^\*$ 取自**最早**快照，可远大于 $\mu_s$，且 $M^\*-\mu_s\le\kappa\cdot\max_i(s-t_i)$（**最大**时龄，非平均）。要使 $L_t-\Delta\le\mu_s$ 成立，需 $\Delta\ge L_t-\mu_s$，而最坏情形 $L_t-\mu_s$ 可达 $\kappa\cdot\max$-age。由于加权**平均**时龄 $<$ **最大**时龄，$\Delta_t^{\text{mean}}$ 一般**不足**，故 $L_t-\Delta_t^{\text{mean}}$ **不是**严格 $1-\delta$ 下界。这与实验中 $\varepsilon=0$ 轻度反保守一致。

### 3c 聚合"最大时龄"inflation — **Status: PROVABLE AS STATED（保守）**
取 $\Delta_t^{\max}=\kappa\cdot\max\{\,s-t_i: w_i>0\,\}$。则 $L_t-\Delta_t^{\max}$ 是 $\mu_s$ 的 anytime-valid 下界。
**证明.** 设 $S$ 为权重支撑样本集（$w_i>0$）。仅由这些样本构成 $K_t$，故其上鞅性要求 $m\ge M^\*_S:=\max_{i\in S}\mu_i$；由命题1式论证 $\Pr(\exists t:M^\*_S<L_t)\le\delta$。又 (A-drift) 给 $M^\*_S\le\mu_s+\kappa\max_{i\in S}(s-t_i)=\mu_s+\Delta_t^{\max}$。于是高概率 $L_t\le M^\*_S\le\mu_s+\Delta_t^{\max}$，即 $L_t-\Delta_t^{\max}\le\mu_s$。∎

**结论与建议.** 命题3a（逐样本平移）是**严格且最紧**的有效构造，应作为论文主定理与实现首选；3c（最大时龄）严格但保守；3b（当前实现的平均时龄）**不严格**，仅当强 recency 加权使权重集中、平均时龄$\approx$最大有效时龄时近似成立——这正解释了实证里 $\gamma=0.04$ 下 SAVED 在漂移格子有效、却在 $\varepsilon=0$ 轻度反保守。**实现应改为 3a。**

---

## 命题4（工作负载级同时覆盖，alpha-spending）
**Status: PROVABLE AS STATED.**

**陈述.** 自适应、可选停时的查询流 $q=1,2,\dots$，分析者可依历史数据自适应选择下一个 estimand。给查询 $q$ 预先（在抽取该查询认证标注前，$\mathcal F$-可预测地）分配 $\delta_q\ge0$ 且 $\sum_q\delta_q\le\delta$。每条用命题1/3a 在水平 $\delta_q$ 的 anytime-valid 下界 $L^{(q)}$。则
$$\Pr\big(\exists q\ \text{被认证}:\ \mathrm{recall}_q<L^{(q)}\big)\le\delta.$$

**证明.** 设 $A_q=\{$查询 $q$ 在其（可选）停时报告的下界破坏其 estimand$\}$。由命题1/3a（对固定 estimand 的 anytime-valid 性，已涵盖该查询内部的可选停时），$\Pr(A_q\mid\mathcal F_{\text{pre}(q)})\le\delta_q$，其中 $\delta_q$ 可预测，故 $\Pr(A_q)\le\delta_q$。由 Boole 不等式（无需独立性，自适应选择 estimand 不影响）
$$\Pr\Big(\bigcup_q A_q\Big)\le\sum_q\Pr(A_q)\le\sum_q\delta_q\le\delta.\ \blacksquare$$

**为何优于 online-FDR.** "保证系统"语义要求**所有**报告下界同时成立（FWER/同时覆盖）；online-FDR 仅控制被认证集合中**假保证的期望比例**（容忍 $\delta$ 比例的假保证），对"保证"是错误语义。Alpha-spending 以 Boole 即得更强的同时覆盖。
*Open risk.* 无界查询流上 $\sum\delta_q\le\delta$ 迫使 $\delta_q\to0$ → 功效衰减；有界 $N$ 适用，无界需改 alpha-investing（保留可信但允许在"已认证且事后验证"时返还预算）。

---

## 汇总
| 命题 | 状态 | 备注 |
|---|---|---|
| 1 无漂移 anytime CS | 严格 close | 含 empirical-Bernstein 可预测 λ |
| 2 加权下注 | 严格 close | $\lambda_i'=w_i\lambda_i$ |
| 3a 逐样本平移 | **严格 close（主定理）** | 仅用上侧漂移界；最紧 |
| 3b 平均时龄 inflation（现实现） | **不严格** | 一般不足，需改 3a |
| 3c 最大时龄 inflation | 严格 close（保守） | 备选 |
| 4 alpha-spending 同时覆盖 | 严格 close | 有界 N；无界需 alpha-investing |

**对实现/论文的行动项.** (i) 把 `SAVEDDriftRobust`/`certify_weighted` 的聚合 $\Delta^{\text{mean}}$ 改为命题3a 的逐样本平移 $m_i=\min(m+\kappa(s-t_i),1)$（严格且更紧）；(ii) 论文主定理用 3a，附录给 3c 对照、3b 反例；(iii) 命题4 注明有界 N。

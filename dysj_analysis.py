import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 核心分析类 
class SurveyBiasAnalysis:
    def __init__(self):
        self.data = None
        self.results = {}
        self.indicator_calculations = {}
    # 1. 加载数据
    def load_data(self, excel_file_path,
                  sheet_names=('入户', '电话', '网络')):
        print("=== 数据加载阶段 ===")
        try:
            df_face = pd.read_excel(excel_file_path, sheet_name=sheet_names[0])
            df_phone = pd.read_excel(excel_file_path, sheet_name=sheet_names[1])
            df_online = pd.read_excel(excel_file_path, sheet_name=sheet_names[2])
        except Exception as e:
            raise RuntimeError(f"读取Excel失败: {e}")
        # 统一列名
        std_cols = {
            'Q1\u200eaqg': 'sf',
            'Q4（shcemtd）': 'ags',
            'Q3（zzzxmyd）': 'cce',
            'Q6（zfgzmyd）': 'les',
            'S2年龄': 'age',
            'S3城乡': 'urban_rural',
            'P1性别': 'gender',
            'P2受教育程度': 'education'
        }
        for df, name in zip((df_face, df_phone, df_online), ('入户', '电话', '网络')):
            df.rename(columns={k: v for k, v in std_cols.items() if k in df.columns}, inplace=True)
            df['survey_type'] = name

        self.data = pd.concat([df_face, df_phone, df_online], ignore_index=True, sort=False)
        print(f"合并完成，总记录数: {len(self.data)}")
        return self.data

    # 2. 数据预处理 
    def preprocess_data(self):
        print("\n=== 数据预处理 ===")
        # 缺失值：只删核心题
        core = ['sf', 'ags', 'cce',
                'les']
        self.data = self.data.dropna(subset=core)
        print(f"删除缺失后记录数: {len(self.data)}")

        # 年龄映射
        age_map = {
            '16-25岁': 20, '26-30岁': 28, '31-40岁': 35, '41-50岁': 45,
            '51-60岁': 55, '61-70岁': 65, '71岁及以上': 75
        }
        self.data['age_numeric'] = self.data['age'].map(age_map)

        # 城乡/教育/性别 简单数值化
        self.data['is_urban'] = self.data['urban_rural'].map({'城镇': 1, '农村': 0})
        self.data['edu_level'] = self.data['education'].map(
            {'未上过学': 1, '小学': 2, '初中': 3, '高中（中专/职高/技校）': 4,
             '大学专科': 5, '大学本科及以上': 6})
        self.data['gender_num'] = self.data['gender'].map({'男': 1, '女': 2})
        print("预处理完成")
        return self.data

    # 3. 计算指标
    def calculate_indicators(self):
        print("计算主要指标")
        # 正向回答
        self.data['s_pos'] = self.data['ssf'].isin([1, 2])
        self.data['g_pos'] = self.data['ags'].isin([1, 2])
        self.data['c_pos'] = self.data['cce'].isin([1, 2])
        self.data['l_pos'] = self.data['les'].isin([1, 2])
        # 不了解
        self.data['g_unk'] = self.data['ags'] == 5
        self.data['c_unk'] = self.data['cce'] == 5
        self.data['l_unk'] = self.data['les'] == 5

        def rate_pos(df, pos_col, unk_col=None):
            if unk_col is not None:
                valid = len(df) - df[unk_col].sum()
                return (df[pos_col].sum() / valid * 100) if valid else 0
            return df[pos_col].mean() * 100

        calc = {
            'qzaqg': self.data.groupby('survey_type').apply(
                lambda d: rate_pos(d, 's_pos')),
            'shcemyd': self.data.groupby('survey_type').apply(
                lambda d: rate_pos(d, 'g_pos', 'g_unk')),
            'zzzxmyd': self.data.groupby('survy_type').apply(
                lambda d: rate_pos(d, 'c_pos', 'c_unk')),
            'zfgzmyd': self.data.groupby('survey_type').apply(
                lambda d: rate_pos(d, 'l_pos', 'l_unk'))
        }
        self.indicator_calculations = calc
        for k, v in calc.items():
            print(f"{k}:\n{v}\n")
        return calc

    #  4. 描述性分析，绘图
    def descriptive_analysis(self):
        print("描述性分析：")
        # 主要指标柱状图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        for ax, (title, ser) in zip(axes, self.indicator_calculations.items()):
            ser.plot(kind='bar', ax=ax, rot=0)
            ax.set_title(title)
            ax.set_ylabel('百分比 (%)')
        plt.tight_layout()
        plt.savefig('三种方式主要指标对比.png', dpi=300)
        plt.show()

    # ---------------- 5. 偏差系数 ----------------
    def calculate_bias_coefficients(self, benchmark='入户'):
        print(f"\n=== 偏差系数 (基准: {benchmark}) ===")
        bias_results = {}
        for name, ser in self.indicator_calculations.items():
            if benchmark not in ser:
                print(f"基准 {benchmark} 缺失，跳过 {name}")
                continue
            base = ser[benchmark]
            tmp = {}
            for mth, val in ser.items():
                if mth == benchmark:
                    continue
                coeff = (val - base) / base * 100
                tmp[mth] = {'偏差%': coeff, '绝对差': val - base}
                direction = '高估' if coeff > 0 else '低估'
                print(f"{name} | {mth} vs {benchmark}: {direction} {abs(coeff):.2f}%")
            bias_results[name] = tmp
        self.results['bias'] = bias_results
        return bias_results

    # ---------------- 6. 方差分析 ----------------
    def anova_analysis(self):
        print("\n=== 方差分析 (ANOVA) ===")
        anova_results = {}

        # 对每个指标进行ANOVA
        indicators = {
            'sf': 'qzaqg',
            'ags': 'shcemyd',
            'cce': 'zzzxmyd',
            'les': 'zfgzmyd'
        }

        for col, name in indicators.items():
            print(f"\n--- {name} ANOVA ---")

            # 数据清洗和验证
            valid_data = self.data[[col, 'survey_type']].dropna()

            # 检查每组数据是否足够
            group_counts = valid_data.groupby('survey_type').size()
            if len(group_counts) < 2 or any(group_counts < 2):
                print(f" {name}: 数据不足，跳过ANOVA")
                continue

            groups = [group[col].values for name, group in valid_data.groupby('survey_type')]

            # 检查方差是否为0
            variances = [np.var(group) for group in groups]
            if any(v == 0 for v in variances):
                print(f" {name}: 存在零方差组，跳过ANOVA")
                continue

            try:
                # 单因素方差分析
                f_stat, p_value = f_oneway(*groups)

                # 处理极小的p值
                if p_value < 1e-300:  # 如果p值过小，设为最小值
                    p_value = 1e-300
                    p_display = "< 1e-300"
                else:
                    p_display = f"{p_value:.4e}"

                print(f"F统计量: {f_stat:.4f}, p值: {p_display}")

                if p_value < 0.05:
                    print("不同调查方式间存在显著差异")
                else:
                    print("不同调查方式间无显著差异")

                # 计算效应量 (eta平方)
                ss_between = 0
                ss_total = 0
                grand_mean = valid_data[col].mean()

                for group in groups:
                    group_mean = np.mean(group)
                    ss_between += len(group) * (group_mean - grand_mean) ** 2
                    ss_total += np.sum((group - grand_mean) ** 2)

                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                print(f"效应量 (η²): {eta_squared:.4f}")

                anova_results[name] = {
                    'F': f_stat,
                    'p': p_value,
                    'p_display': p_display,
                    'eta_squared': eta_squared,
                    'significant': p_value < 0.05
                }

                # 事后检验 (Tukey HSD)
                self._posthoc_tukey(col, name)

            except Exception as e:
                print(f" {name} ANOVA计算失败: {e}")
                anova_results[name] = {
                    'F': 0, 'p': 1, 'p_display': '计算错误',
                    'eta_squared': 0, 'significant': False
                }

        self.results['anova'] = anova_results
        return anova_results

    def _posthoc_tukey(self, col, name):
        """Tukey HSD事后检验"""
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        try:
            tukey = pairwise_tukeyhsd(
                endog=self.data[col],
                groups=self.data['survey_type'],
                alpha=0.05
            )
            print(f"Tukey HSD事后检验 - {name}:")
            print(tukey)
        except Exception as e:
            print(f"Tukey检验失败: {e}")

    # ---------------- 7. 误差来源三维度分析 ----------------
    def three_dimension_analysis(self):
        print("\n三维度误差来源分析")
        # 量化分析各维度影响
        dimension_impact = {}

        # 1. 调查主体维度 - 社会期许偏差验证
        print("\n1. 调查主体维度分析:")
        subject_scores = self._calculate_subject_bias()
        dimension_impact['调查主体'] = subject_scores

        # 2. 调查过程维度 - 数据质量指标
        print("\n2. 调查过程维度分析:")
        process_scores = self._calculate_process_quality()
        dimension_impact['调查过程'] = process_scores

        # 3. 调查技术维度 - 样本代表性
        print("\n3. 调查技术维度分析:")
        tech_scores = self._calculate_tech_representativeness()
        dimension_impact['调查技术'] = tech_scores

        self.results['dimension_analysis'] = {
            'error_types': error_types,
            'impact_scores': dimension_impact
        }

        return dimension_impact

    def _calculate_subject_bias(self):
        """计算调查主体维度的社会期许偏差"""
        # 基于敏感问题回答差异估算社会期许偏差
        sensitive_indicators = ['ags', 'les']
        bias_scores = {}

        for method in ['入户', '电话', '网络']:
            method_data = self.data[self.data['survey_type'] == method]
            positivity_rate = method_data[sensitive_indicators].isin([1, 2]).mean().mean()
            bias_scores[method] = positivity_rate * 100

        print(f"社会期许偏差得分: {bias_scores}")
        return bias_scores

    def _calculate_process_quality(self):
        """计算调查过程维度的数据质量"""
        quality_scores = {}

        for method in ['入户', '电话', '网络']:
            method_data = self.data[self.data['survey_type'] == method]

            # 数据完整性 (非缺失率)
            completeness = 1 - method_data.isnull().mean().mean()

            # 回答一致性 (方差倒数)
            consistency = 1 / (method_data['sf'].std() + 0.001)

            # 极端回答比例
            extreme_ratio = (method_data['sf'].isin([1, 5])).mean()

            quality_score = (completeness * 0.4 + consistency * 0.3 + (1 - extreme_ratio) * 0.3) * 100
            quality_scores[method] = quality_score

        print(f"过程质量得分: {quality_scores}")
        return quality_scores

    def _calculate_tech_representativeness(self):
        """计算调查技术维度的样本代表性 
        rep_scores = {}

        # 基准分布 (基于国家统计局数据调整)
        benchmark_dist = {
            'urban_rural': 0.65,  # 城镇比例（根据第七次人口普查）
            'edu_high': 0.25,  # 高等教育比例（大专及以上）
            'age_median': 45  # 中位年龄参考
        }

        for method in ['入户', '电话', '网络']:
            method_data = self.data[self.data['survey_type'] == method]

            # 检查数据有效性
            if len(method_data) == 0:
                rep_scores[method] = 50  # 默认中等分数
                continue

            scores = []

            # 1. 城乡代表性
            try:
                urban_rate = method_data['is_urban'].mean()
                if not np.isnan(urban_rate):
                    urban_score = 1 - abs(urban_rate - benchmark_dist['urban_rural'])
                    scores.append(max(0, urban_score))
            except:
                pass

            # 2. 教育代表性
            try:
                edu_high_rate = (method_data['edu_level'] >= 4).mean()  # 高中及以上
                if not np.isnan(edu_high_rate):
                    edu_score = 1 - abs(edu_high_rate - benchmark_dist['edu_high'])
                    scores.append(max(0, edu_score))
            except:
                pass

            # 3. 年龄分布代表性
            try:
                age_data = method_data['age_numeric'].dropna()
                if len(age_data) > 0:
                    # 使用年龄分布的多样性作为代表性指标
                    age_diversity = 1 - (age_data.std() / 30)  # 标准化
                    age_score = max(0, min(1, age_diversity))
                    scores.append(age_score)
            except:
                pass

            # 计算平均得分
            if len(scores) > 0:
                rep_score = np.mean(scores) * 100
            else:
                rep_score = 50  # 默认分数

            rep_scores[method] = rep_score

        print(f"技术代表性得分: {rep_scores}")
        return rep_scores

    # ---------------- 8. 偏差影响量化 ----------------
    def quantify_bias_impact(self):
        """偏差影响程度量化 - 
        print("\n偏差影响程度量化")

        impact_results = {}

        for indicator, values_series in self.indicator_calculations.items():
            try:
                # 确保处理的是数值数组
                values_array = np.array([float(x) for x in values_series.values])

                max_val = np.max(values_array)
                min_val = np.min(values_array)
                range_val = max_val - min_val
                mean_val = np.mean(values_array)

                if mean_val > 0:
                    cv = (np.std(values_array) / mean_val) * 100
                    impact_score = (range_val / mean_val) * 100
                else:
                    cv = 0
                    impact_score = 0

                # 确定影响等级
                if impact_score > 15:
                    level = '极高'
                elif impact_score > 10:
                    level = '高'
                elif impact_score > 5:
                    level = '中'
                else:
                    level = '低'

                impact_results[indicator] = {
                    '极差': range_val,
                    '变异系数%': cv,
                    '影响程度%': impact_score,
                    '影响等级': level
                }

                print(f"{indicator}: 极差={range_val:.2f}%, 变异系数={cv:.2f}%, 影响程度={impact_score:.2f}% ({level})")

            except Exception as e:
                print(f"计算{indicator}时出错: {e}")
                impact_results[indicator] = {
                    '极差': 0, '变异系数%': 0, '影响程度%': 0, '影响等级': '未知'
                }

        self.results['bias_impact'] = impact_results
        return impact_results
    # ---------------- 综合报告 ----------------
    def comprehensive_report(self):
        print("\n" + "=" * 80)
        print("                  qzaqg调查偏差量化分析报告")
        print("=" * 80)
        print(f"总样本: {len(self.data)}")
        print("样本分布:")
        print(self.data['survey_type'].value_counts())

        print("\n主要指标（%）:")
        for name, ser in self.indicator_calculations.items():
            print(f"\n{name}:")
            for m, v in ser.items():
                print(f"  {m}: {v:.2f}%")

        print("\n偏差分析:")
        for name, biases in self.results.get('bias', {}).items():
            print(f"\n{name}:")
            for m, info in biases.items():
                direction = "高估" if info['偏差%'] > 0 else "低估"
                print(f"  {m}: {info['偏差%']:+.2f}% ({direction})")

        print("\n方差分析结果:")
        for name, result in self.results.get('anova', {}).items():
            sig_flag = "显著" if result['significant'] else "不显著"
            p_display = result.get('p_display', f"{result['p']:.4e}")
            print(f"  {name}: F={result['F']:.3f}, p={p_display}, η²={result['eta_squared']:.4f} {sig_flag}")

        print("\n偏差影响程度:")
        for name, impact in self.results.get('bias_impact', {}).items():
            print(f"  {name}: {impact['影响程度%']:.2f}% ({impact['影响等级']}影响)")
        print("\n优化建议：")
        print("1. 建立多模式混合调查框架，校正电话/网络高估倾向")
        print("2. 针对敏感指标，采用面访为主、其他方式为辅")
        print("3. 开发基于三维度分析的偏差校正模型")
        print("4. 加强访问员培训，减少社会期许偏差")
        print("5. 优化问卷设计，降低题目理解偏差")


# ---------- 主函数 ----------
def main():
    analyzer = SurveyBiasAnalysis()
    file_path = input('请输入Excel完整路径：').strip().strip('"')

    # 执行完整分析流程
    analyzer.load_data(file_path)
    analyzer.preprocess_data()
    analyzer.calculate_indicators()
    analyzer.descriptive_analysis()
    analyzer.calculate_bias_coefficients()
    analyzer.anova_analysis()
    analyzer.three_dimension_analysis()
    analyzer.quantify_bias_impact()
    analyzer.comprehensive_report()


if __name__ == '__main__':
    main()

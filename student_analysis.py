import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# 设置图表样式先于字体设置
sns.set_style("whitegrid")

# 设置中文字体 - 改进版本
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

class StudentDataAnalyzer:
    def __init__(self, csv_path):
        """初始化分析器并加载数据"""
        self.df = pd.read_csv(csv_path)
        self.prepare_data()
    
    def prepare_data(self):
        """数据预处理"""
        # 检查和清理数据
        print("正在处理数据...")
        
        # 处理可能的缺失值和数据类型问题
        self.df = self.df.dropna()
        
        # 确保数值列是正确的数据类型
        numeric_columns = ['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
                          'attendance_percentage', 'sleep_hours', 'exercise_frequency', 
                          'mental_health_rating', 'exam_score']
        
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # 再次清理转换后的缺失值
        self.df = self.df.dropna()
        
        # 重命名列名为中文便于理解
        self.column_mapping = {
            'student_id': '学生ID',
            'age': '年龄',
            'gender': '性别',
            'study_hours_per_day': '每日学习小时数',
            'social_media_hours': '每日社交媒体小时数',
            'netflix_hours': '每日Netflix小时数',
            'part_time_job': '是否有兼职',
            'attendance_percentage': '出勤率',
            'sleep_hours': '每日睡眠小时数',
            'diet_quality': '饮食质量',
            'exercise_frequency': '运动频率',
            'parental_education_level': '父母教育水平',
            'internet_quality': '网络质量',
            'mental_health_rating': '心理健康评分',
            'extracurricular_participation': '是否参与课外活动',
            'exam_score': '学业表现分数'
        }
        
        print("数据基本信息：")
        print(f"数据行数: {len(self.df)}")
        print(f"数据列数: {len(self.df.columns)}")
        print("\n数据前5行：")
        print(self.df.head())
    
    def plot_performance_distribution(self):
        """一、学业表现分数的直方图"""
        plt.figure(figsize=(12, 8))
        
        # 绘制直方图
        n, bins, patches = plt.hist(self.df['exam_score'], bins=30, alpha=0.7, 
                                   color='skyblue', edgecolor='black', linewidth=1)
        
        # 设置标题和标签
        plt.title('学业表现分数分布直方图', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('学业表现分数', fontsize=14)
        plt.ylabel('学生人数', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_score = self.df['exam_score'].mean()
        std_score = self.df['exam_score'].std()
        median_score = self.df['exam_score'].median()
        
        plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, 
                   label=f'平均分: {mean_score:.1f}')
        plt.axvline(median_score, color='green', linestyle='--', linewidth=2, 
                   label=f'中位数: {median_score:.1f}')
        
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()
        
        print(f"\n学业表现统计信息：")
        print(f"平均分: {mean_score:.2f}")
        print(f"中位数: {median_score:.2f}")
        print(f"标准差: {std_score:.2f}")
        print(f"最高分: {self.df['exam_score'].max():.2f}")
        print(f"最低分: {self.df['exam_score'].min():.2f}")
    
    def plot_numerical_relationships(self):
        """二、探索关键数值型习惯与学业表现的关系"""
        numerical_vars = [
            ('study_hours_per_day', '每日学习小时数'),
            ('attendance_percentage', '出勤率'),
            ('sleep_hours', '每日睡眠小时数'),
            ('social_media_hours', '每日社交媒体小时数')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('数值型习惯与学业表现分数的关系', fontsize=18, fontweight='bold')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (var, chinese_name) in enumerate(numerical_vars):
            row = i // 2
            col = i % 2
            
            # 绘制散点图
            axes[row, col].scatter(
                self.df[var],
                self.df['exam_score'],
                alpha=0.6,
                s=50,
                color=colors[i],
                edgecolors='black',
                linewidth=0.5,
                label='数据点'  # 修改为中文label
            )
            
            # 添加趋势线
            z = np.polyfit(self.df[var], self.df['exam_score'], 1)
            p = np.poly1d(z)
            axes[row, col].plot(self.df[var], p(self.df[var]), "r--", alpha=0.8, linewidth=2, label='趋势线') # 新增label
            
            axes[row, col].set_xlabel(chinese_name, fontsize=12)
            axes[row, col].set_ylabel('学业表现分数', fontsize=12)
            axes[row, col].set_title(f'{chinese_name} vs 学业表现分数', fontsize=14, fontweight='bold')
            axes[row, col].grid(True, alpha=0.3)
            
            # 计算相关系数
            correlation = self.df[var].corr(self.df['exam_score'])
            axes[row, col].text(0.05, 0.95, f'相关系数: {correlation:.3f}', 
                              transform=axes[row, col].transAxes, 
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                              fontsize=11, fontweight='bold')
            
            axes[row, col].legend(loc='upper left')  # 显示图例
        
        plt.tight_layout()
        plt.show()
    
    def plot_categorical_relationships(self):
        """三、探索关键类别型习惯与学业表现的关系"""
        categorical_vars = [
            ('part_time_job', '是否有兼职'),
            ('diet_quality', '饮食质量'),
            ('parental_education_level', '父母教育水平'),
            ('extracurricular_participation', '是否参与课外活动')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('类别型习惯与学业表现分数的关系', fontsize=18, fontweight='bold')
        
        for i, (var, chinese_name) in enumerate(categorical_vars):
            row = i // 2
            col = i % 2
            
            # 安全地处理类别数据
            try:
                # 移除缺失值并转换为字符串
                clean_data = self.df[[var, 'exam_score']].dropna()
                clean_data[var] = clean_data[var].astype(str)
                
                # 获取唯一值并排序
                unique_values = sorted(clean_data[var].unique())
                data_for_box = [clean_data[clean_data[var] == val]['exam_score'].values 
                              for val in unique_values]
                
                # 创建箱形图
                box_plot = axes[row, col].boxplot(data_for_box, labels=unique_values, 
                                                patch_artist=True, showmeans=True,
                                                meanprops={"marker":"o","markerfacecolor":"white", 
                                                         "markeredgecolor":"black","markersize":8})
                
                # 美化箱形图
                colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
                for patch, color in zip(box_plot['boxes'], colors[:len(unique_values)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                axes[row, col].set_xlabel(chinese_name, fontsize=12)
                axes[row, col].set_ylabel('学业表现分数', fontsize=12)
                axes[row, col].set_title(f'{chinese_name} vs 学业表现分数', fontsize=14, fontweight='bold')
                axes[row, col].grid(True, alpha=0.3)
                
                # 添加平均值标注
                for j, val in enumerate(unique_values):
                    subset = clean_data[clean_data[var] == val]['exam_score']
                    mean_val = subset.mean()
                    count = len(subset)
                    axes[row, col].text(j+1, mean_val + 2, f'均值: {mean_val:.1f}\n(n={count})', 
                                      ha='center', va='bottom', fontweight='bold',
                                      bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
                
            except Exception as e:
                print(f"处理变量 {var} 时出错: {e}")
                axes[row, col].text(0.5, 0.5, f'数据处理错误\n{var}', 
                                  transform=axes[row, col].transAxes, 
                                  ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self):
        """四、数值型变量的相关性热力图"""
        # 选择数值型变量
        numerical_columns = [
            'age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
            'attendance_percentage', 'sleep_hours', 'exercise_frequency', 
            'mental_health_rating', 'exam_score'
        ]
        
        # 过滤存在的列
        available_columns = [col for col in numerical_columns if col in self.df.columns]
        
        # 创建中文标签映射
        chinese_labels_map = {
            'age': '年龄',
            'study_hours_per_day': '每日学习小时数',
            'social_media_hours': '每日社交媒体小时数',
            'netflix_hours': '每日Netflix小时数',
            'attendance_percentage': '出勤率',
            'sleep_hours': '每日睡眠小时数',
            'exercise_frequency': '运动频率',
            'mental_health_rating': '心理健康评分',
            'exam_score': '学业表现分数'
        }
        
        chinese_labels = [chinese_labels_map[col] for col in available_columns]
        
        # 计算相关矩阵
        correlation_matrix = self.df[available_columns].corr()
        
        # 创建热力图
        fig = plt.figure(figsize=(14, 12)) # 获取figure对象
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        heatmap = sns.heatmap(
            correlation_matrix, 
            mask=mask,
            annot=True, 
            cmap='RdYlBu_r', 
            center=0,
            square=True,
            fmt='.3f',
            cbar_kws={"shrink": .8},
            linewidths=0.5
        )
        
        # 为colorbar添加标签
        cbar = heatmap.collections[0].colorbar
        cbar.set_label('相关系数(越高越相关)', rotation=270, labelpad=15)
        
        # 设置中文标签
        heatmap.set_xticklabels(chinese_labels, rotation=45, ha='right', fontsize=10)
        heatmap.set_yticklabels(chinese_labels, rotation=0, fontsize=10)
        
        plt.title('数值型变量相关性热力图', fontsize=18, fontweight='bold', pad=20)
        
        # 调整子图参数，为标题和x轴标签留出空间
        plt.subplots_adjust(top=0.95, bottom=0.20) # 增加顶部和底部边距
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # 可以在tight_layout中进一步微调
        plt.show()
        
        # 输出与学业表现最相关的因素
        if 'exam_score' in available_columns:
            performance_corr = correlation_matrix['exam_score'].drop('exam_score').sort_values(key=abs, ascending=False)
            print("\n与学业表现分数相关性排序（按绝对值）：")
            for i, (var, corr) in enumerate(performance_corr.items()):
                chinese_name = chinese_labels_map[var]
                print(f"{i+1}. {chinese_name}: {corr:.3f}")
    
    def generate_summary_report(self):
        """生成分析总结报告"""
        print("\n" + "="*60)
        print("学生习惯与学业表现分析报告")
        print("="*60)
        
        # 基本统计
        print(f"\n1. 数据概览:")
        print(f"   - 样本数量: {len(self.df)} 名学生")
        print(f"   - 平均年龄: {self.df['age'].mean():.1f} 岁")
        print(f"   - 平均学业表现分数: {self.df['exam_score'].mean():.1f}")
        
        # 性别分布
        if 'gender' in self.df.columns:
            gender_dist = self.df['gender'].value_counts()
            print(f"\n2. 性别分布:")
            for gender, count in gender_dist.items():
                print(f"   - {gender}: {count} 人 ({count/len(self.df)*100:.1f}%)")
        
        # 关键发现
        print(f"\n3. 关键发现:")
        
        # 学习时间与成绩的关系
        study_corr = self.df['study_hours_per_day'].corr(self.df['exam_score'])
        print(f"   - 每日学习时间与学业表现的相关性: {study_corr:.3f}")
        
        # 出勤率与成绩的关系
        attendance_corr = self.df['attendance_percentage'].corr(self.df['exam_score'])
        print(f"   - 出勤率与学业表现的相关性: {attendance_corr:.3f}")
        
        # 社交媒体使用与成绩的关系
        social_corr = self.df['social_media_hours'].corr(self.df['exam_score'])
        print(f"   - 社交媒体使用时间与学业表现的相关性: {social_corr:.3f}")
        
        # 兼职工作的影响
        if 'part_time_job' in self.df.columns:
            job_effect = self.df.groupby('part_time_job')['exam_score'].mean()
            if 'Yes' in job_effect.index:
                print(f"   - 有兼职学生平均分: {job_effect['Yes']:.1f}")
            if 'No' in job_effect.index:
                print(f"   - 无兼职学生平均分: {job_effect['No']:.1f}")
        
        print(f"\n4. 建议:")
        if study_corr > 0.3:
            print(f"   - 学习时间与成绩呈中等正相关，建议增加有效学习时间")
        if attendance_corr > 0.3:
            print(f"   - 出勤率对成绩有积极影响，建议保持良好出勤")
        if social_corr < -0.2:
            print(f"   - 过度使用社交媒体可能影响学习，建议合理控制使用时间")
    
    def run_complete_analysis(self):
        """运行完整的分析流程"""
        print("开始学生习惯与学业表现分析...")
        print("\n正在生成可视化图表...")
        
        try:
            # 1. 学业表现分布
            print("\n1. 绘制学业表现分数分布图...")
            self.plot_performance_distribution()
            
            # 2. 数值型变量关系
            print("\n2. 分析数值型习惯与学业表现的关系...")
            self.plot_numerical_relationships()
            
            # 3. 类别型变量关系
            print("\n3. 分析类别型习惯与学业表现的关系...")
            self.plot_categorical_relationships()
            
            # 4. 相关性热力图
            print("\n4. 绘制相关性热力图...")
            self.plot_correlation_heatmap()
            
            # 5. 生成总结报告
            print("\n5. 生成分析报告...")
            self.generate_summary_report()
            
            print("\n分析完成！所有图表已显示。")
            
        except Exception as e:
            print(f"分析过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    # 数据文件路径
    csv_path = r"e:\desktop\student_habits_performance.csv"
    
    try:
        # 创建分析器实例
        analyzer = StudentDataAnalyzer(csv_path)
        
        # 运行完整分析
        analyzer.run_complete_analysis()
        
    except FileNotFoundError:
        print(f"错误：找不到数据文件 {csv_path}")
        print("请确保文件路径正确且文件存在。")
    except Exception as e:
        print(f"程序启动时出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

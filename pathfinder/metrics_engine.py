#!/usr/bin/env python3
"""
Advanced Metrics Engine for Career Path Analysis
Processes cleaned job data to generate comprehensive career metrics.

This script calculates detailed metrics across multiple dimensions:
- Job Market metrics (volume, accessibility, remote options)
- Compensation analysis (salary ranges, median pay)
- Competition & Accessibility (skill barriers, entry requirements)
- Skill Compatibility (tools, learning curves)
- Future Forecast placeholders (for trend analysis)

Author: AI Assistant
Created: 2025
"""

import csv
import json
import statistics
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple


class CareerMetricsEngine:
    """Advanced metrics calculation engine for career path analysis."""
    
    def __init__(self):
        """Initialize the metrics engine."""
        self.data_dir = Path("pathfinder/data")
        self.output_dir = Path("pathfinder/output")
        self.career_data = {}
        self.all_skills = Counter()
        self.global_stats = {}
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("🔧 Advanced Career Metrics Engine")
        print("=" * 50)
    
    def load_cleaned_data(self) -> bool:
        """Load all cleaned CSV files."""
        print("📥 Loading cleaned job data...")
        
        if not self.data_dir.exists():
            print(f"❌ Data directory not found: {self.data_dir}")
            return False
        
        # Find all cleaned CSV files
        cleaned_files = list(self.data_dir.glob("cleaned_*.csv"))
        
        if not cleaned_files:
            print(f"❌ No cleaned CSV files found in {self.data_dir}")
            return False
        
        total_jobs = 0
        
        for file_path in cleaned_files:
            # Extract career path name from filename
            career_name = self._extract_career_name(file_path.name)
            
            try:
                # Load CSV data
                jobs = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    jobs = list(reader)
                
                if jobs:
                    self.career_data[career_name] = jobs
                    total_jobs += len(jobs)
                    print(f"  ✓ {career_name}: {len(jobs)} jobs")
                    
                    # Collect all skills for global analysis
                    for job in jobs:
                        skills_str = job.get('extracted_skills', '')
                        if skills_str:
                            skills = [s.strip() for s in skills_str.split(',') if s.strip()]
                            self.all_skills.update(skills)
                else:
                    print(f"  ⚠️ {career_name}: No data found")
                    
            except Exception as e:
                print(f"  ❌ Error loading {file_path.name}: {e}")
        
        print(f"\n📊 Total jobs loaded: {total_jobs}")
        print(f"📈 Career paths found: {len(self.career_data)}")
        print(f"🔧 Unique skills identified: {len(self.all_skills)}")
        
        return len(self.career_data) > 0
    
    def _extract_career_name(self, filename: str) -> str:
        """Extract career name from cleaned CSV filename."""
        # Remove 'cleaned_' prefix and '.csv' suffix
        name = filename.replace('cleaned_', '').replace('.csv', '')
        
        # Convert underscores to spaces and title case
        name = name.replace('_', ' ').title()
        
        # Handle special cases for better readability
        name = name.replace('Ai ', 'AI ')
        name = name.replace('Nlp ', 'NLP ')
        name = name.replace('Bi ', 'BI ')
        
        return name
    
    def calculate_global_stats(self):
        """Calculate global statistics for normalization."""
        print("\n🌍 Calculating global statistics...")
        
        all_jobs = []
        for jobs in self.career_data.values():
            all_jobs.extend(jobs)
        
        # Salary statistics
        salaries = []
        for job in all_jobs:
            try:
                if job.get('avg_salary') and float(job['avg_salary']) > 0:
                    salaries.append(float(job['avg_salary']))
            except (ValueError, TypeError):
                continue
        
        # Entry-level statistics
        entry_level_jobs = sum(1 for job in all_jobs if job.get('is_entry_level') == 'True')
        
        # Remote work statistics
        remote_jobs = sum(1 for job in all_jobs if job.get('is_remote') == 'True')
        
        # Skills statistics
        skill_counts = []
        for job in all_jobs:
            try:
                count = int(job.get('skills_count', 0))
                skill_counts.append(count)
            except (ValueError, TypeError):
                skill_counts.append(0)
        
        self.global_stats = {
            'total_jobs': len(all_jobs),
            'salary_min': min(salaries) if salaries else 0,
            'salary_max': max(salaries) if salaries else 0,
            'salary_median': statistics.median(salaries) if salaries else 0,
            'entry_level_rate': (entry_level_jobs / len(all_jobs)) * 100 if all_jobs else 0,
            'remote_rate': (remote_jobs / len(all_jobs)) * 100 if all_jobs else 0,
            'avg_skills_per_job': statistics.mean(skill_counts) if skill_counts else 0,
            'top_skills': dict(self.all_skills.most_common(20))
        }
        
        print(f"  📊 Global salary range: ${self.global_stats['salary_min']:,.0f} - ${self.global_stats['salary_max']:,.0f}")
        print(f"  🎓 Global entry-level rate: {self.global_stats['entry_level_rate']:.1f}%")
        print(f"  🏠 Global remote rate: {self.global_stats['remote_rate']:.1f}%")
        print(f"  🔧 Avg skills per job: {self.global_stats['avg_skills_per_job']:.1f}")
    
    def calculate_career_metrics(self, career_name: str, jobs: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a single career path."""
        
        # Basic job market metrics
        total_jobs = len(jobs)
        
        # Entry-level analysis
        entry_jobs = sum(1 for job in jobs if job.get('is_entry_level') == 'True')
        entry_pct = (entry_jobs / total_jobs) * 100 if total_jobs > 0 else 0
        
        # Remote work analysis
        remote_jobs = sum(1 for job in jobs if job.get('is_remote') == 'True')
        remote_pct = (remote_jobs / total_jobs) * 100 if total_jobs > 0 else 0
        
        # Salary analysis
        salaries = []
        salary_ranges = []
        for job in jobs:
            try:
                avg_sal = float(job.get('avg_salary', 0))
                if avg_sal > 0:
                    salaries.append(avg_sal)
                
                min_sal = float(job.get('min_salary', 0))
                max_sal = float(job.get('max_salary', 0))
                if min_sal > 0 and max_sal > 0:
                    salary_ranges.append(max_sal - min_sal)
            except (ValueError, TypeError):
                continue
        
        median_salary = statistics.median(salaries) if salaries else 0
        avg_salary_range = statistics.mean(salary_ranges) if salary_ranges else 0
        
        # Skill analysis
        career_skills = Counter()
        skill_counts = []
        for job in jobs:
            try:
                count = int(job.get('skills_count', 0))
                skill_counts.append(count)
            except (ValueError, TypeError):
                skill_counts.append(0)
            
            skills_str = job.get('extracted_skills', '')
            if skills_str:
                skills = [s.strip() for s in skills_str.split(',') if s.strip()]
                career_skills.update(skills)
        
        avg_skills = statistics.mean(skill_counts) if skill_counts else 0
        top_skills = dict(career_skills.most_common(10))
        
        # Junior/entry detection from titles
        junior_titles = 0
        for job in jobs:
            title = job.get('title', '').lower()
            if re.search(r'\b(junior|entry|graduate|trainee|intern|associate)\b', title):
                junior_titles += 1
        
        junior_title_pct = (junior_titles / total_jobs) * 100 if total_jobs > 0 else 0
        
        # Competition analysis (skill barrier)
        skill_barrier_score = avg_skills  # Higher skills = higher barrier
        
        # Calculate skill overlap with global top skills
        global_top_skills = set(self.global_stats.get('top_skills', {}).keys())
        career_skill_set = set(career_skills.keys())
        skill_overlap = len(career_skill_set.intersection(global_top_skills))
        skill_overlap_pct = (skill_overlap / len(global_top_skills)) * 100 if global_top_skills else 0
        
        # Calculate relative positions
        global_median_salary = self.global_stats.get('salary_median', 0)
        salary_premium = ((median_salary - global_median_salary) / global_median_salary * 100) if global_median_salary > 0 else 0
        
        global_entry_rate = self.global_stats.get('entry_level_rate', 0)
        entry_accessibility = entry_pct - global_entry_rate  # Positive = more accessible
        
        global_remote_rate = self.global_stats.get('remote_rate', 0)
        remote_advantage = remote_pct - global_remote_rate  # Positive = more remote-friendly
        
        return {
            "career_path": career_name,
            
            # 🧑‍💻 Job Market Metrics
            "M1_job_postings": total_jobs,
            "M2_entry_pct": round(entry_pct, 1),
            "M3_remote_pct": round(remote_pct, 1),
            "M4_junior_title_pct": round(junior_title_pct, 1),
            "M5_market_share": round((total_jobs / self.global_stats['total_jobs']) * 100, 1),
            
            # 💵 Compensation Metrics
            "C1_salary_median": round(median_salary, 0),
            "C2_salary_range_avg": round(avg_salary_range, 0),
            "C3_salary_premium_pct": round(salary_premium, 1),
            "C4_compensation_rank": 0,  # Will be calculated later
            
            # ⚔️ Competition & Accessibility Metrics
            "S1_skill_barrier_score": round(skill_barrier_score, 1),
            "S2_entry_accessibility": round(entry_accessibility, 1),
            "S3_competition_index": round(100 - entry_pct, 1),  # Higher = more competitive
            
            # 🔧 Skill Compatibility Metrics
            "K1_avg_skills_required": round(avg_skills, 1),
            "K2_skill_overlap_pct": round(skill_overlap_pct, 1),
            "K3_unique_skills_count": len(career_skills),
            "K4_top_skills": top_skills,
            
            # 🚀 Market Position Metrics
            "F1_remote_advantage": round(remote_advantage, 1),
            "F2_growth_potential": round(avg_skills * 10, 1),  # Skills as growth proxy
            "F3_versatility_score": round(skill_overlap_pct + (avg_skills * 5), 1),
            
            # 📊 Raw Counts for Reference
            "raw_entry_jobs": entry_jobs,
            "raw_remote_jobs": remote_jobs,
            "raw_salary_count": len(salaries),
            "raw_skills_total": sum(career_skills.values())
        }
    
    def calculate_all_metrics(self) -> List[Dict[str, Any]]:
        """Calculate metrics for all career paths."""
        print("\n🔢 Calculating comprehensive metrics...")
        
        all_metrics = []
        
        for career_name, jobs in self.career_data.items():
            print(f"  📊 Processing {career_name}...")
            metrics = self.calculate_career_metrics(career_name, jobs)
            all_metrics.append(metrics)
        
        # Calculate compensation ranks
        all_metrics.sort(key=lambda x: x['C1_salary_median'], reverse=True)
        for i, metrics in enumerate(all_metrics, 1):
            metrics['C4_compensation_rank'] = i
        
        # Sort back to original order (by career path name)
        all_metrics.sort(key=lambda x: x['career_path'])
        
        return all_metrics
    
    def save_metrics_csv(self, metrics: List[Dict[str, Any]], filename: str = "full_metrics_raw.csv") -> str:
        """Save metrics to CSV file."""
        output_path = self.output_dir / filename
        
        try:
            # Prepare data for CSV (exclude complex nested data)
            csv_metrics = []
            for metric in metrics:
                csv_row = {}
                for key, value in metric.items():
                    if key == 'K4_top_skills':
                        # Convert top skills to JSON string
                        csv_row[key] = json.dumps(value)
                    else:
                        csv_row[key] = value
                csv_metrics.append(csv_row)
            
            # Write CSV
            if csv_metrics:
                fieldnames = csv_metrics[0].keys()
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(csv_metrics)
            
            print(f"💾 Metrics saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Error saving metrics CSV: {e}")
            return ""
    
    def save_detailed_analysis(self, metrics: List[Dict[str, Any]], filename: str = "metrics_analysis.json") -> str:
        """Save detailed analysis including nested data."""
        output_path = self.output_dir / filename
        
        try:
            analysis_data = {
                'global_stats': self.global_stats,
                'career_metrics': metrics,
                'metadata': {
                    'total_careers': len(metrics),
                    'total_jobs': self.global_stats.get('total_jobs', 0),
                    'analysis_date': str(Path().resolve()),
                    'top_global_skills': list(self.all_skills.most_common(20))
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Detailed analysis saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Error saving detailed analysis: {e}")
            return ""
    
    def print_metrics_summary(self, metrics: List[Dict[str, Any]]):
        """Print a summary of key metrics."""
        print("\n" + "="*70)
        print("📊 COMPREHENSIVE CAREER METRICS SUMMARY")
        print("="*70)
        
        print(f"\n🌍 Global Market Overview:")
        print(f"   • Total Jobs Analyzed: {self.global_stats['total_jobs']:,}")
        print(f"   • Salary Range: ${self.global_stats['salary_min']:,.0f} - ${self.global_stats['salary_max']:,.0f}")
        print(f"   • Global Entry-Level Rate: {self.global_stats['entry_level_rate']:.1f}%")
        print(f"   • Global Remote Rate: {self.global_stats['remote_rate']:.1f}%")
        print(f"   • Average Skills per Job: {self.global_stats['avg_skills_per_job']:.1f}")
        
        print(f"\n🎯 Career Path Analysis:")
        print()
        
        # Table header
        header = "| Career Path              | Jobs | Salary   | Entry% | Remote% | Skills | Rank |"
        separator = "|--------------------------|------|----------|--------|---------|--------|------|"
        
        print(header)
        print(separator)
        
        # Sort by compensation rank for display
        display_metrics = sorted(metrics, key=lambda x: x['C4_compensation_rank'])
        
        for m in display_metrics:
            career_name = m['career_path'][:24]  # Truncate long names
            jobs = m['M1_job_postings']
            salary = m['C1_salary_median']
            entry = m['M2_entry_pct']
            remote = m['M3_remote_pct']
            skills = m['K1_avg_skills_required']
            rank = m['C4_compensation_rank']
            
            row = f"| {career_name:<24} | {jobs:4d} | ${salary:7.0f} | {entry:5.1f}% | {remote:6.1f}% | {skills:6.1f} | {rank:4d} |"
            print(row)
        
        print()
        
        # Key insights
        highest_salary = max(metrics, key=lambda x: x['C1_salary_median'])
        most_accessible = max(metrics, key=lambda x: x['M2_entry_pct'])
        most_remote = max(metrics, key=lambda x: x['M3_remote_pct'])
        highest_skills = max(metrics, key=lambda x: x['K1_avg_skills_required'])
        
        print(f"💰 **Highest Salary**: {highest_salary['career_path']} (${highest_salary['C1_salary_median']:,.0f})")
        print(f"🎓 **Most Accessible**: {most_accessible['career_path']} ({most_accessible['M2_entry_pct']:.1f}% entry-level)")
        print(f"🏠 **Most Remote-Friendly**: {most_remote['career_path']} ({most_remote['M3_remote_pct']:.1f}% remote)")
        print(f"🧠 **Highest Skill Requirements**: {highest_skills['career_path']} ({highest_skills['K1_avg_skills_required']:.1f} skills avg)")
        
        print("="*70)
    
    def run_analysis(self) -> bool:
        """Run the complete metrics analysis."""
        print("🚀 Starting advanced metrics analysis...\n")
        
        # Load data
        if not self.load_cleaned_data():
            return False
        
        # Calculate global statistics
        self.calculate_global_stats()
        
        # Calculate career metrics
        metrics = self.calculate_all_metrics()
        
        # Save results
        csv_path = self.save_metrics_csv(metrics)
        json_path = self.save_detailed_analysis(metrics)
        
        # Display summary
        self.print_metrics_summary(metrics)
        
        if csv_path and json_path:
            print(f"\n💾 Results saved to:")
            print(f"   📊 CSV: {csv_path}")
            print(f"   📋 JSON: {json_path}")
        
        print(f"\n🎉 Advanced metrics analysis completed successfully!")
        
        return True


def main():
    """Main function to run the metrics engine."""
    try:
        engine = CareerMetricsEngine()
        success = engine.run_analysis()
        
        if not success:
            print("❌ Analysis failed. Please check your data files and try again.")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python3
"""
Advanced Career Pathfinder Dashboard - Simplified Version
Interactive Streamlit dashboard for comprehensive career path analysis.

This version works without pandas dependency issues by using the CSV data
directly and implementing the analysis logic in pure Python.

Author: AI Assistant
Created: 2025
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import csv
import json
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="🧭 Advanced Career Pathfinder",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Advanced Career Pathfinder - Data-driven career analysis with 20+ metrics"
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #ff6b6b;
        margin: 0.5rem 0;
    }
    .top-career {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #4CAF50;
    }
    .dimension-score {
        font-size: 1.2em;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_metrics_data():
    """Load the advanced metrics data from CSV using pure Python."""
    metrics_file = Path("pathfinder/output/full_metrics_raw.csv")
    
    if not metrics_file.exists():
        st.error(f"❌ Metrics file not found: {metrics_file}")
        st.error("Please run `python3 pathfinder/metrics_engine.py` first to generate the metrics data.")
        st.stop()
    
    try:
        data = []
        with open(metrics_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric columns
                for key, value in row.items():
                    if key != 'career_path' and key != 'K4_top_skills':
                        try:
                            row[key] = float(value)
                        except (ValueError, TypeError):
                            pass
                data.append(row)
        return data
    except Exception as e:
        st.error(f"❌ Error loading metrics data: {e}")
        st.stop()


def calculate_dimension_scores(data):
    """Calculate average scores for each dimension using prefix matching."""
    dimensions = {}
    
    for career in data:
        career_name = career['career_path']
        
        # Job Market (M-series) - normalize percentages and counts
        m_metrics = []
        for key, value in career.items():
            if key.startswith('M') and key != 'M5_market_share' and isinstance(value, (int, float)):
                if 'pct' in key:
                    m_metrics.append(value / 100)  # Convert percentages to 0-1
                elif 'postings' in key:
                    m_metrics.append(value / 50)  # Normalize job counts (max ~50)
                else:
                    m_metrics.append(value)
        
        # Compensation (C-series) - normalize salary values
        c_metrics = []
        salary_values = [career.get('C1_salary_median', 0)]
        salary_min, salary_max = 85000, 170000  # Approximate range from data
        for key, value in career.items():
            if key.startswith('C') and 'rank' not in key and isinstance(value, (int, float)):
                if 'salary' in key:
                    normalized = (value - salary_min) / (salary_max - salary_min)
                    c_metrics.append(max(0, min(1, normalized)))
                elif 'premium' in key:
                    c_metrics.append((value + 50) / 100)  # Convert premium to 0-1 scale
        
        # Accessibility (S-series) - invert barriers for accessibility
        s_metrics = []
        for key, value in career.items():
            if key.startswith('S') and isinstance(value, (int, float)):
                if 'barrier' in key or 'competition' in key:
                    s_metrics.append(1 - (value / 100))  # Invert for accessibility
                elif 'accessibility' in key:
                    s_metrics.append((value + 10) / 20)  # Normalize accessibility score
        
        # Skill Compatibility (K-series)
        k_metrics = []
        for key, value in career.items():
            if key.startswith('K') and key != 'K4_top_skills' and isinstance(value, (int, float)):
                if 'overlap' in key:
                    k_metrics.append(value / 100)  # Convert percentage to 0-1
                else:
                    k_metrics.append(value / 5)  # Normalize skill counts
        
        # Future Forecast (F-series)
        f_metrics = []
        for key, value in career.items():
            if key.startswith('F') and isinstance(value, (int, float)):
                f_metrics.append((value + 10) / 20)  # Normalize to 0-1 range
        
        # Calculate averages
        dimensions[career_name] = {
            'M': sum(m_metrics) / len(m_metrics) if m_metrics else 0.5,
            'C': sum(c_metrics) / len(c_metrics) if c_metrics else 0.5,
            'S': sum(s_metrics) / len(s_metrics) if s_metrics else 0.5,
            'K': sum(k_metrics) / len(k_metrics) if k_metrics else 0.5,
            'F': sum(f_metrics) / len(f_metrics) if f_metrics else 0.5
        }
    
    return dimensions


def create_radar_plot(career_data, career_name):
    """Create a radar plot for a specific career path."""
    categories = ['Job Market', 'Compensation', 'Accessibility', 'Skill Compatibility', 'Future Forecast']
    values = [
        career_data.get('M', 0),
        career_data.get('C', 0), 
        career_data.get('S', 0),
        career_data.get('K', 0),
        career_data.get('F', 0)
    ]
    
    # Close the radar plot
    values += values[:1]
    categories += categories[:1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=career_name,
        line_color='rgb(102, 126, 234)',
        fillcolor='rgba(102, 126, 234, 0.25)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=['0', '0.2', '0.4', '0.6', '0.8', '1.0']
            )
        ),
        showlegend=False,
        title=f"Multi-Dimensional Analysis: {career_name}",
        title_x=0.5
    )
    
    return fig


def create_salary_chart(data):
    """Create a salary comparison chart using pure Plotly Graph Objects."""
    career_names = [row['career_path'] for row in data]
    salaries = [row.get('C1_salary_median', 0) for row in data]
    
    # Sort by salary for better visualization
    sorted_data = sorted(zip(career_names, salaries), key=lambda x: x[1])
    sorted_careers, sorted_salaries = zip(*sorted_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sorted_salaries,
        y=sorted_careers,
        orientation='h',
        marker=dict(
            color=sorted_salaries,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Salary ($)")
        ),
        text=[f"${salary:,.0f}" for salary in sorted_salaries],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Salary Comparison by Career Path",
        title_x=0.5,
        xaxis_title="Median Salary ($)",
        yaxis_title="Career Path",
        height=500,
        margin=dict(l=200)  # More space for career path names
    )
    
    return fig


def create_entry_level_chart(data):
    """Create an entry-level opportunities chart using pure Plotly Graph Objects."""
    career_names = [row['career_path'] for row in data]
    entry_pcts = [row.get('M2_entry_pct', 0) for row in data]
    
    # Sort by entry-level percentage
    sorted_data = sorted(zip(career_names, entry_pcts), key=lambda x: x[1])
    sorted_careers, sorted_entry = zip(*sorted_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sorted_entry,
        y=sorted_careers,
        orientation='h',
        marker=dict(
            color=sorted_entry,
            colorscale='Greens',
            showscale=True,
            colorbar=dict(title="Entry-Level %")
        ),
        text=[f"{pct:.1f}%" for pct in sorted_entry],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Entry-Level Opportunities by Career Path",
        title_x=0.5,
        xaxis_title="Entry-Level Percentage (%)",
        yaxis_title="Career Path",
        height=500,
        margin=dict(l=200)  # More space for career path names
    )
    
    return fig


def main():
    """Main dashboard function."""
    
    # Header
    st.title("🧭 Advanced Career Pathfinder Dashboard")
    st.markdown("""
    **Interactive analysis of data career paths using 20+ comprehensive metrics**
    
    Adjust the dimension weights below to customize the analysis for your priorities.
    """)
    
    # Load data
    data = load_metrics_data()
    
    # Calculate dimension scores
    dimension_scores = calculate_dimension_scores(data)
    
    # Sidebar for controls
    st.sidebar.header("🎛️ Analysis Controls")
    st.sidebar.markdown("Adjust weights to match your career priorities:")
    
    # Weight sliders
    weights = {}
    
    weights['M'] = st.sidebar.slider(
        "🧑‍💻 Job Market Weight", 
        min_value=0.0, max_value=1.0, value=0.2, step=0.05,
        help="Job availability, entry-level opportunities, remote options"
    )
    
    weights['C'] = st.sidebar.slider(
        "💵 Compensation Weight", 
        min_value=0.0, max_value=1.0, value=0.3, step=0.05,
        help="Salary levels, ranges, and premiums"
    )
    
    weights['S'] = st.sidebar.slider(
        "⚔️ Accessibility Weight", 
        min_value=0.0, max_value=1.0, value=0.2, step=0.05,
        help="Competition levels and entry barriers"
    )
    
    weights['K'] = st.sidebar.slider(
        "🔧 Skill Compatibility Weight", 
        min_value=0.0, max_value=1.0, value=0.2, step=0.05,
        help="Required skills and market overlap"
    )
    
    weights['F'] = st.sidebar.slider(
        "🚀 Future Forecast Weight", 
        min_value=0.0, max_value=1.0, value=0.1, step=0.05,
        help="Growth potential and market positioning"
    )
    
    total_weight = sum(weights.values())
    
    # Normalize weights
    if total_weight > 0:
        for key in weights:
            weights[key] = weights[key] / total_weight
    
    st.sidebar.markdown(f"**Total Weight:** {total_weight:.2f}")
    if total_weight < 0.95 or total_weight > 1.05:
        st.sidebar.warning("⚠️ Consider adjusting weights to sum to ~1.0")
    
    # Calculate weighted scores
    ranked_careers = []
    for career in data:
        career_name = career['career_path']
        dims = dimension_scores.get(career_name, {})
        
        weighted_score = (
            dims.get('M', 0.5) * weights['M'] +
            dims.get('C', 0.5) * weights['C'] +
            dims.get('S', 0.5) * weights['S'] +
            dims.get('K', 0.5) * weights['K'] +
            dims.get('F', 0.5) * weights['F']
        )
        
        ranked_careers.append({
            'career_path': career_name,
            'weighted_score': weighted_score,
            'dimensions': dims,
            'raw_data': career
        })
    
    # Sort by score
    ranked_careers.sort(key=lambda x: x['weighted_score'], reverse=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🏆 Career Path Rankings")
        
        # Display ranking table
        table_data = []
        for i, career in enumerate(ranked_careers, 1):
            dims = career['dimensions']
            raw = career['raw_data']
            table_data.append([
                i,
                career['career_path'],
                f"{career['weighted_score']:.3f}",
                f"{dims.get('M', 0):.2f}",
                f"{dims.get('C', 0):.2f}",
                f"{dims.get('S', 0):.2f}",
                f"{dims.get('K', 0):.2f}",
                f"{dims.get('F', 0):.2f}",
                f"${raw.get('C1_salary_median', 0):,.0f}"
            ])
        
        # Create and display table
        table_html = """
        <table style="width:100%; border-collapse: collapse;">
        <tr style="background-color: #f0f2f6;">
            <th>Rank</th><th>Career Path</th><th>Score</th><th>Market</th><th>Compensation</th><th>Access</th><th>Skills</th><th>Future</th><th>Salary</th>
        </tr>
        """
        
        for row in table_data:
            rank_style = ""
            if row[0] == 1:
                rank_style = "background-color: #fff3cd;"
            elif row[0] == 2:
                rank_style = "background-color: #d4e6f1;"
            elif row[0] == 3:
                rank_style = "background-color: #fadbd8;"
            
            table_html += f"<tr style='{rank_style}'>"
            for cell in row:
                table_html += f"<td style='padding: 8px; border: 1px solid #ddd;'>{cell}</td>"
            table_html += "</tr>"
        
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🎯 Top Recommendation")
        
        if ranked_careers:
            top_career = ranked_careers[0]
            raw = top_career['raw_data']
            
            st.markdown(f"""
            <div class="top-career">
            <h3>{top_career['career_path']}</h3>
            <p><strong>Score:</strong> {top_career['weighted_score']:.3f}</p>
            <p><strong>Salary:</strong> ${raw.get('C1_salary_median', 0):,.0f}</p>
            <p><strong>Entry-Level:</strong> {raw.get('M2_entry_pct', 0):.1f}%</p>
            <p><strong>Remote Options:</strong> {raw.get('M3_remote_pct', 0):.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Dimension breakdown
            dims = top_career['dimensions']
            st.markdown("**Dimension Scores:**")
            for dim, label in [('M', 'Market'), ('C', 'Compensation'), ('S', 'Accessibility'), ('K', 'Skills'), ('F', 'Future')]:
                score = dims.get(dim, 0)
                st.markdown(f"• **{label}**: {score:.2f}")
    
    # Career selection for detailed analysis
    st.subheader("🔍 Detailed Career Analysis")
    
    career_names = [career['career_path'] for career in ranked_careers]
    selected_career = st.selectbox(
        "Select a career path for detailed analysis:",
        options=career_names,
        index=0
    )
    
    # Find selected career data
    selected_data = None
    for career in ranked_careers:
        if career['career_path'] == selected_career:
            selected_data = career
            break
    
    if selected_data:
        col1, col2 = st.columns(2)
        
        with col1:
            # Create radar plot
            radar_fig = create_radar_plot(selected_data['dimensions'], selected_career)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        with col2:
            raw = selected_data['raw_data']
            st.markdown("**Key Metrics:**")
            st.metric("Jobs Available", f"{int(raw.get('M1_job_postings', 0)):,}")
            st.metric("Median Salary", f"${raw.get('C1_salary_median', 0):,.0f}")
            st.metric("Entry-Level Rate", f"{raw.get('M2_entry_pct', 0):.1f}%")
            st.metric("Remote Work Rate", f"{raw.get('M3_remote_pct', 0):.1f}%")
            st.metric("Skill Requirements", f"{raw.get('K1_avg_skills_required', 0):.1f}")
            st.metric("Salary Premium", f"{raw.get('C3_salary_premium_pct', 0):.1f}%")
    
    # Visualizations
    st.subheader("📊 Market Analysis")
    
    tab1, tab2 = st.tabs(["💰 Salary Analysis", "🎓 Entry-Level Analysis"])
    
    with tab1:
        salary_fig = create_salary_chart(data)
        st.plotly_chart(salary_fig, use_container_width=True)
    
    with tab2:
        entry_fig = create_entry_level_chart(data)
        st.plotly_chart(entry_fig, use_container_width=True)
    
    # Key insights
    st.subheader("💡 Key Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("**Market Leaders:**")
        
        highest_salary = max(data, key=lambda x: x.get('C1_salary_median', 0))
        most_accessible = max(data, key=lambda x: x.get('M2_entry_pct', 0))
        most_remote = max(data, key=lambda x: x.get('M3_remote_pct', 0))
        
        st.markdown(f"💰 **Highest Salary**: {highest_salary['career_path']} (${highest_salary.get('C1_salary_median', 0):,.0f})")
        st.markdown(f"🎓 **Most Accessible**: {most_accessible['career_path']} ({most_accessible.get('M2_entry_pct', 0):.1f}% entry-level)")
        st.markdown(f"🏠 **Most Remote**: {most_remote['career_path']} ({most_remote.get('M3_remote_pct', 0):.1f}% remote)")
    
    with insights_col2:
        st.markdown("**Your Custom Ranking:**")
        for i, career in enumerate(ranked_careers[:3]):
            rank_emoji = ["🥇", "🥈", "🥉"][i]
            st.markdown(f"{rank_emoji} **{career['career_path']}** (Score: {career['weighted_score']:.3f})")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>🧭 Advanced Career Pathfinder</strong> | Built with real job market data</p>
    <p><em>Analysis based on 286 real job postings • 20+ comprehensive metrics • 5 analytical dimensions</em></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as sm
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Load data with caching for better performance
@st.cache_data
def load_data():
    return pd.read_csv('short_wide_xWS2_df.csv')

# Cache model fitting for better performance
@st.cache_data
def fit_models(df):
    models = {}
    for i in range(1, 11):
        mp_col = f'MP{i}' if i <= 5 else 'MP5'  # Fix the MP column issue for models 6-10
        formula = f"xWS2r{i} ~ RankAdjEM+I(RankAdjEM**2)+I(RankAdjEM**3)"
        models[f'r{i}mod3'] = sm.ols(formula=formula, data=df[df[mp_col] > 100]).fit()
    return models

df = load_data()
models = fit_models(df)

# Create VWS functions using the models
def create_vws_functions(models):
    vws_functions = {}
    for i in range(1, 11):
        model = models[f'r{i}mod3']
        def make_vws_func(model):
            return lambda rank: (
                model.params['Intercept'] + 
                model.params['RankAdjEM'] * rank + 
                model.params['I(RankAdjEM ** 2)'] * (rank**2) + 
                model.params['I(RankAdjEM ** 3)'] * (rank**3)
            )
        vws_functions[f'vws{i}'] = make_vws_func(model)
    return vws_functions

vws_functions = create_vws_functions(models)

def get_team_bin(rank):
    """Determine team bin based on rank"""
    if rank <= 20:
        return "1-20"
    elif rank <= 40:
        return "21-40"
    elif rank <= 60:
        return "41-60"
    elif rank <= 80:
        return "61-80"
    elif rank <= 100:
        return "81-100"
    elif rank <= 120:
        return "101-120"
    elif rank <= 140:
        return "121-140"
    elif rank <= 160:
        return "141-160"
    elif rank <= 180:
        return "161-180"
    elif rank <= 200:
        return "181-200"
    elif rank <= 220:
        return "201-220"
    elif rank <= 240:
        return "221-240"
    elif rank <= 260:
        return "241-260"
    elif rank <= 280:
        return "261-280"
    elif rank <= 300:
        return "281-300"
    elif rank <= 320:
        return "301-320"
    elif rank <= 340:
        return "321-340"
    else:
        return "341+"

def estimate_team_rank_from_vws(total_vws, replacement_value=0):
    """Estimate team rank based on total VWS using the same models but in reverse"""
    # If total VWS is very low, return a high rank
    if total_vws <= 0:
        return 350
    
    # Create a function that calculates total VWS for a given rank
    def get_total_vws_for_rank(rank):
        total = 0
        for i in range(1, 11):
            raw_vws = vws_functions[f'vws{i}'](rank)
            adjusted_vws = max(0, raw_vws - replacement_value)
            total += adjusted_vws
        return total
    
    # Use binary search to find the rank that produces the closest total VWS
    left_rank, right_rank = 1, 350
    best_rank = 175
    best_diff = float('inf')
    
    # Binary search for the best match
    for _ in range(20):  # Limit iterations to prevent infinite loops
        mid_rank = (left_rank + right_rank) / 2
        mid_vws = get_total_vws_for_rank(mid_rank)
        diff = abs(mid_vws - total_vws)
        
        if diff < best_diff:
            best_diff = diff
            best_rank = mid_rank
        
        if mid_vws > total_vws:
            # VWS is too high, need higher (worse) rank
            left_rank = mid_rank
        else:
            # VWS is too low, need lower (better) rank
            right_rank = mid_rank
        
        # Stop if we're close enough
        if abs(left_rank - right_rank) < 0.1:
            break
    
    return int(round(best_rank))

def create_histogram(team_rank, player_rank, bins=30):
    """Create a histogram with highlighted target value"""
    # Validation
    if team_rank < 1:
        raise ValueError("Team rank too low")
    if not (1 <= player_rank <= 10):
        raise ValueError("Invalid player rank")
    
    # Get data
    column = f'xWS2r{player_rank}'
    mins_col = f'MP{player_rank}' if player_rank <= 5 else 'MP5'
    tm_bin = get_team_bin(team_rank)
    
    # Filter data
    data = df[(df[mins_col] >= 100) & (df['KP_Bins_20'] == tm_bin)][column].dropna()
    
    if len(data) == 0:
        raise ValueError(f"No valid data found for the specified parameters")
    
    # Calculate target value
    target_value = np.round(vws_functions[f'vws{player_rank}'](team_rank), 2)
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n, bin_edges, patches = ax.hist(data, bins=bins, alpha=0.7, 
                                   color='skyblue', edgecolor='black', linewidth=0.5)
    
    # Find and highlight target bin
    bin_index = np.digitize(target_value, bin_edges) - 1
    bin_index = max(0, min(bin_index, len(patches) - 1))
    
    patches[bin_index].set_facecolor('red')
    patches[bin_index].set_alpha(0.8)
    
    # Add vertical line at target value
    ax.axvline(target_value, color='black', linestyle='--', 
              linewidth=2, alpha=0.8, label=f'Target: {target_value}')
    ax.legend()
    
    # Labels and formatting
    ax.set_xlabel('VWS')
    ax.set_ylabel('Frequency')
    ax.set_title(f'VWS Distribution for Player Rank {player_rank} on Team Rank {team_rank}\n(Target: {target_value})')
    ax.grid(True, alpha=0.3)
    
    return fig, ax

def calculate_vws_metrics(team_rank, replacement_value=0):
    """Calculate VWS metrics for all player positions"""
    vws_values = []
    for i in range(1, 11):
        raw_vws = vws_functions[f'vws{i}'](team_rank)
        # Subtract replacement value and ensure no negative values
        adjusted_vws = max(0, raw_vws - replacement_value)
        vws_values.append(adjusted_vws)
    
    sum_total = sum(vws_values)
    # Handle case where sum_total is 0 (all values became 0)
    if sum_total == 0:
        percentages = [0] * 10
    else:
        percentages = [(vws / sum_total) * 100 for vws in vws_values]
    
    return percentages, vws_values

def create_plotly_pie_chart(values, labels, title, color_sequence=None):
    """Create a Plotly pie chart"""
    fig = px.pie(
        values=values, 
        names=labels, 
        title=title,
        color_discrete_sequence=color_sequence or px.colors.qualitative.Set3
    )
    fig.update_traces(
        textposition='inside', 
        textinfo='percent',
        hovertemplate='<b>%{label}</b><br>Value: %{value:.2f}<br>Percent: %{percent}<extra></extra>'
    )
    fig.update_layout(
        showlegend=False,
        height=200,
        margin=dict(t=30, b=0, l=0, r=0),
        font=dict(size=10)
    )
    return fig

def create_matplotlib_pie_chart(values, labels, colors):
    """Create a Matplotlib pie chart"""
    fig, ax = plt.subplots(figsize=(3, 3))
    wedges, texts, autotexts = ax.pie(
        values, 
        labels=labels, 
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    
    # Improve text appearance
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(8)
    
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig

def display_vws_dashboard(team_rank, replacement_value=0, use_plotly=True):
    """Display the VWS dashboard with metrics and pie charts"""
    percentages, vws_values = calculate_vws_metrics(team_rank, replacement_value)
    
    st.header("📊 Virtual Win Shares Dashboard")
    
    # Display total VWS and replacement info
    total_vws = sum(vws_values)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Team VWS (Above Replacement)", f"{total_vws:.2f}")
    with col2:
        if replacement_value > 0:
            st.metric("Replacement Level VWS Value", f"{replacement_value:.2f}", 
                     help="Value subtracted from each player representing easily replaceable production")
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
              '#DDA0DD', '#98D8E8', '#F7DC6F', '#BB8FCE', '#85C1E9']
    
    # First row - Players 1-5
    st.subheader("Players 1-5")
    cols_row1 = st.columns(5)
    
    for i in range(5):
        with cols_row1[i]:
            st.metric(
                label=f"VWS {i+1}", 
                value=f"{percentages[i]:.1f}%",
                delta=f"{vws_values[i]:.2f}",
                help=f"Virtual Win Shares above replacement for player position {i+1}"
            )
            
            # Create pie chart
            if use_plotly:
                fig = create_plotly_pie_chart(
                    values=[percentages[i], 100-percentages[i]], 
                    labels=[f'VWS {i+1}', 'Other'],
                    title=f"Player {i+1}",
                    color_sequence=[colors[i], '#E8E8E8']
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = create_matplotlib_pie_chart(
                    values=[percentages[i], 100-percentages[i]],
                    labels=[f'VWS {i+1}', 'Other'],
                    colors=[colors[i], '#E8E8E8']
                )
                st.pyplot(fig)
    
    # Second row - Players 6-10
    st.subheader("Players 6-10")
    cols_row2 = st.columns(5)
    
    for i in range(5, 10):
        with cols_row2[i-5]:
            st.metric(
                label=f"VWS {i+1}", 
                value=f"{percentages[i]:.1f}%",
                delta=f"{vws_values[i]:.2f}",
                help=f"Virtual Win Shares above replacement for player position {i+1}"
            )
            
            # Create pie chart
            if use_plotly:
                fig = create_plotly_pie_chart(
                    values=[percentages[i], 100-percentages[i]], 
                    labels=[f'VWS {i+1}', 'Other'],
                    title=f"Player {i+1}",
                    color_sequence=[colors[i], '#E8E8E8']
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = create_matplotlib_pie_chart(
                    values=[percentages[i], 100-percentages[i]],
                    labels=[f'VWS {i+1}', 'Other'],
                    colors=[colors[i], '#E8E8E8']
                )
                st.pyplot(fig)
    
    return dict(zip([f'VWS{i+1}' for i in range(10)], percentages))

def create_editable_summary_table(team_rank, replacement_value):
    """Create an editable summary table with real-time projections"""
    st.subheader("📝 Interactive VWS Editor")
    st.markdown("*Edit VWS values below to see projected team rank and percentage changes*")
    
    # Get initial values
    percentages, vws_values = calculate_vws_metrics(team_rank, replacement_value)
    
    # Create input fields for each position
    st.markdown("**Edit VWS Above Replacement Values:**")
    cols = st.columns(5)
    edited_vws = []
    
    # First row (player ranks 1-5)
    for i in range(5):
        with cols[i]:
            edited_value = st.number_input(
                f"Player Rank {i+1}",
                min_value=0.0,
                max_value=30.0,
                value=float(vws_values[i]),
                step=0.1,
                format="%.2f",
                key=f"vws_edit_{i+1}"
            )
            edited_vws.append(edited_value)
    
    # Second row (player ranks 6-10)
    cols2 = st.columns(5)
    for i in range(5, 10):
        with cols2[i-5]:
            edited_value = st.number_input(
                f"Player Rank {i+1}",
                min_value=0.0,
                max_value=30.0,
                value=float(vws_values[i]),
                step=0.1,
                format="%.2f",
                key=f"vws_edit_{i+1}"
            )
            edited_vws.append(edited_value)
    
    # Calculate new metrics
    edited_total = sum(edited_vws)
    if edited_total > 0:
        edited_percentages = [(vws / edited_total) * 100 for vws in edited_vws]
    else:
        edited_percentages = [0] * 10
    
    # Estimate new team rank
    estimated_rank = estimate_team_rank_from_vws(edited_total, replacement_value)
    
    # Display projections
    st.markdown("---")
    st.subheader("📈 Projections Based on Edited Values")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Projected Total VWS", 
            f"{edited_total:.2f}",
            delta=f"{edited_total - sum(vws_values):.2f}"
        )
    with col2:
        st.metric(
            "Estimated Team Rank", 
            f"{estimated_rank}",
            delta=f"{estimated_rank - team_rank}",
            delta_color="inverse"
        )
    with col3:
        original_total = sum(vws_values)
        pct_change = ((edited_total - original_total) / original_total * 100) if original_total > 0 else 0
        st.metric(
            "VWS Change", 
            f"{pct_change:+.1f}%"
        )
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        'Player Rank': range(1, 11),
        'Original VWS': vws_values,
        'Edited VWS': edited_vws,
        'VWS Change': [edited_vws[i] - vws_values[i] for i in range(10)],
        'Original %': percentages,
        'Edited %': edited_percentages,
        'Percentage Change': [edited_percentages[i] - percentages[i] for i in range(10)]
    })
    
    # Format the dataframe for better display
    styled_df = comparison_df.style.format({
        'Original VWS': '{:.2f}',
        'Edited VWS': '{:.2f}',
        'VWS Change': '{:+.2f}',
        'Original %': '{:.1f}%',
        'Edited %': '{:.1f}%',
        'Percentage Change': '{:+.1f}%'
    }).background_gradient(subset=['VWS Change', 'Percentage Change'], cmap='RdYlGn', vmin=-2, vmax=2)
    
    st.dataframe(styled_df, use_container_width=True)
    
    return edited_vws, edited_percentages, estimated_rank

def get_plot_download_data(fig):
    """Convert matplotlib figure to bytes for download"""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    return img_buffer.getvalue()

def main():
    st.set_page_config(
        page_title="VWS Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🏀 Virtual Win Shares Analysis Dashboard")
    st.markdown("Analyze projected VWS and compare with similarly ranked teams")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Parameters")
        
        team_rank = st.number_input(
            "Team KenPom Rank:",
            min_value=1,
            max_value=350,
            value=50,
            step=1,
            help="Enter the team's KenPom ranking (1-350)"
        )
        
        player_rank = st.number_input(
            "Player Position Rank:",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            help="Enter the player's position rank on the team (1-10)"
        )
        
        replacement_value = st.number_input(
            "Replacement Level VWS Value:",
            min_value=0.0,
            max_value=3.0,
            value=0.0,
            step=0.1,
            help="Value representing easily replaceable player production (0-3). This will be subtracted from each player's VWS."
        )
        
        chart_type = st.radio(
            "Chart Library:",
            ["Plotly (Interactive)", "Matplotlib (Static)"]
        )
        use_plotly = chart_type.startswith("Plotly")
        
        bins = st.slider("Histogram Bins:", 10, 50, 20)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📈 Individual Analysis", "📊 Team Dashboard", "📝 Interactive Editor"])
    
    with tab1:
        st.subheader(f"Individual Player Analysis - Position {player_rank}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            try:
                with st.spinner("Generating histogram..."):
                    fig, ax = create_histogram(team_rank, player_rank, bins=bins)
                st.pyplot(fig)
                
                # Download button
                st.download_button(
                    label="📥 Download Histogram",
                    data=get_plot_download_data(fig),
                    file_name=f"vws_histogram_team{team_rank}_player{player_rank}.png",
                    mime="image/png"
                )
                
            except Exception as e:
                st.error(f"Error generating histogram: {str(e)}")
        
        with col2:
            # Individual metrics
            try:
                vws_value = vws_functions[f'vws{player_rank}'](team_rank)
                st.metric("Projected VWS", f"{vws_value:.3f}")
                
                team_bin = get_team_bin(team_rank)
                st.info(f"Team Bin: {team_bin}")
                
                # Show some statistics
                column = f'xWS2r{player_rank}'
                mins_col = f'MP{player_rank}' if player_rank <= 5 else 'MP5'
                data = df[(df[mins_col] >= 100) & (df['KP_Bins_20'] == team_bin)][column].dropna()
                
                if len(data) > 0:
                    st.write("**Comparison Statistics:**")
                    st.write(f"Mean: {data.mean():.3f}")
                    st.write(f"Median: {data.median():.3f}")
                    st.write(f"Std Dev: {data.std():.3f}")
                    st.write(f"Sample Size: {len(data)}")
                    
                    percentile = (data < vws_value).mean() * 100
                    st.metric("Percentile Rank", f"{percentile:.1f}%")
                
            except Exception as e:
                st.error(f"Error calculating metrics: {str(e)}")
    
    with tab2:
        st.subheader("Complete Team VWS Analysis")
        try:
            metrics = display_vws_dashboard(team_rank, replacement_value, use_plotly)
            
            # Summary statistics
            with st.expander("📈 Summary Statistics"):
                percentages, vws_values = calculate_vws_metrics(team_rank, replacement_value)
                summary_df = pd.DataFrame({
                    'Player Rank': range(1, 11),
                    'VWS Above Replacement': vws_values,
                    'Percentage': percentages
                })
                st.dataframe(summary_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error generating dashboard: {str(e)}")
    
    with tab3:
        try:
            edited_vws, edited_percentages, estimated_rank = create_editable_summary_table(team_rank, replacement_value)
        except Exception as e:
            st.error(f"Error creating interactive editor: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard for analyzing Virtual Win Shares based on KenPom rankings*")

if __name__ == "__main__":
    main()
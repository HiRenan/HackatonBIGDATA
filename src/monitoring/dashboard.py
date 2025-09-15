#!/usr/bin/env python3
"""
Phase 6: Monitoring Dashboard
Enterprise-grade MLflow monitoring dashboard with business metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import mlflow
import mlflow.tracking
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import time
import psutil
import os
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.phase6_config import get_config
from evaluation.metrics import wmape, mape, mae, rmse
from architecture.observers import event_publisher

logger = logging.getLogger(__name__)

class MLflowDashboard:
    """MLflow monitoring dashboard with business metrics"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().dict()
        self.mlflow_client = None
        self.experiment_cache = {}
        self.refresh_interval = self.config.get('monitoring', {}).get('metrics_collection_interval', 60)

    def initialize_mlflow(self):
        """Initialize MLflow connection"""
        try:
            mlflow_config = self.config.get('mlflow', {})
            mlflow.set_tracking_uri(mlflow_config.get('tracking_uri', 'http://localhost:5000'))

            self.mlflow_client = mlflow.tracking.MlflowClient()
            st.success("✅ Connected to MLflow")
            return True

        except Exception as e:
            st.error(f"❌ Failed to connect to MLflow: {e}")
            logger.error(f"MLflow connection failed: {e}")
            return False

    def get_experiments(self) -> List[Dict[str, Any]]:
        """Get all experiments from MLflow"""
        try:
            if not self.mlflow_client:
                return []

            experiments = self.mlflow_client.search_experiments()
            return [
                {
                    'experiment_id': exp.experiment_id,
                    'name': exp.name,
                    'lifecycle_stage': exp.lifecycle_stage,
                    'artifact_location': exp.artifact_location,
                    'creation_time': exp.creation_time,
                    'last_update_time': exp.last_update_time
                }
                for exp in experiments
            ]

        except Exception as e:
            logger.error(f"Failed to get experiments: {e}")
            return []

    def get_runs(self, experiment_id: str, max_results: int = 100) -> pd.DataFrame:
        """Get runs from specific experiment"""
        try:
            if not self.mlflow_client:
                return pd.DataFrame()

            runs = self.mlflow_client.search_runs(
                experiment_ids=[experiment_id],
                max_results=max_results,
                order_by=["start_time DESC"]
            )

            if runs.empty:
                return pd.DataFrame()

            # Add derived metrics
            runs['duration_minutes'] = (runs['end_time'] - runs['start_time']).dt.total_seconds() / 60
            runs['status_icon'] = runs['status'].map({
                'FINISHED': '✅',
                'FAILED': '❌',
                'RUNNING': '🏃',
                'SCHEDULED': '⏰'
            })

            return runs

        except Exception as e:
            logger.error(f"Failed to get runs: {e}")
            return pd.DataFrame()

    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # GPU metrics (if available)
            gpu_metrics = {}
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_metrics = {
                        'gpu_utilization': gpu.load * 100,
                        'gpu_memory_used': gpu.memoryUsed,
                        'gpu_memory_total': gpu.memoryTotal,
                        'gpu_temperature': gpu.temperature
                    }
            except ImportError:
                pass

            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
                **gpu_metrics
            }

        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}

    def render_header(self):
        """Render dashboard header"""
        st.set_page_config(
            page_title="Hackathon Forecast 2025 - Monitoring Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("📊 Hackathon Forecast 2025 - Monitoring Dashboard")
        st.markdown("**Enterprise-grade MLOps monitoring for retail forecasting**")

        # Status indicators
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if self.mlflow_client:
                st.metric("MLflow Status", "✅ Connected", "")
            else:
                st.metric("MLflow Status", "❌ Disconnected", "")

        with col2:
            system_metrics = self.get_system_metrics()
            cpu_usage = system_metrics.get('cpu_percent', 0)
            st.metric("CPU Usage", f"{cpu_usage:.1f}%", "")

        with col3:
            memory_usage = system_metrics.get('memory_percent', 0)
            st.metric("Memory Usage", f"{memory_usage:.1f}%", "")

        with col4:
            current_time = datetime.now().strftime("%H:%M:%S")
            st.metric("Last Update", current_time, "")

    def render_experiment_overview(self):
        """Render experiment overview section"""
        st.header("🧪 Experiment Overview")

        experiments = self.get_experiments()

        if not experiments:
            st.warning("No experiments found. Make sure MLflow is running and experiments exist.")
            return

        # Experiment selector
        exp_names = [exp['name'] for exp in experiments]
        selected_exp_name = st.selectbox("Select Experiment", exp_names)

        selected_exp = next((exp for exp in experiments if exp['name'] == selected_exp_name), None)
        if not selected_exp:
            return

        # Experiment metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Experiment ID", selected_exp['experiment_id'])

        with col2:
            creation_time = datetime.fromtimestamp(selected_exp['creation_time'] / 1000)
            st.metric("Created", creation_time.strftime("%Y-%m-%d"))

        with col3:
            runs_df = self.get_runs(selected_exp['experiment_id'])
            st.metric("Total Runs", len(runs_df))

        with col4:
            if not runs_df.empty:
                successful_runs = len(runs_df[runs_df['status'] == 'FINISHED'])
                success_rate = (successful_runs / len(runs_df)) * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")

        return selected_exp, runs_df

    def render_runs_table(self, runs_df: pd.DataFrame):
        """Render runs table with key metrics"""
        if runs_df.empty:
            st.warning("No runs found for this experiment.")
            return

        st.subheader("📋 Recent Runs")

        # Prepare display columns
        display_columns = ['status_icon', 'run_name', 'start_time', 'duration_minutes']

        # Add metric columns if they exist
        metric_columns = [col for col in runs_df.columns if col.startswith('metrics.')]
        wmape_cols = [col for col in metric_columns if 'wmape' in col.lower()]
        mae_cols = [col for col in metric_columns if 'mae' in col.lower()]

        if wmape_cols:
            display_columns.extend(wmape_cols[:2])  # Show top 2 WMAPE metrics
        if mae_cols:
            display_columns.extend(mae_cols[:2])   # Show top 2 MAE metrics

        # Filter and display
        display_df = runs_df[display_columns].copy()
        display_df['duration_minutes'] = display_df['duration_minutes'].round(2)

        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )

        # Run selector for detailed view
        if st.checkbox("Show detailed run analysis"):
            run_names = runs_df['run_name'].dropna().tolist()
            if run_names:
                selected_run = st.selectbox("Select run for details", run_names)
                selected_run_data = runs_df[runs_df['run_name'] == selected_run].iloc[0]
                self.render_run_details(selected_run_data)

    def render_run_details(self, run_data: pd.Series):
        """Render detailed view of a specific run"""
        st.subheader(f"🔍 Run Details: {run_data['run_name']}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Basic Information**")
            st.write(f"**Status:** {run_data['status_icon']} {run_data['status']}")
            st.write(f"**Start Time:** {run_data['start_time']}")
            st.write(f"**Duration:** {run_data['duration_minutes']:.2f} minutes")

        with col2:
            st.markdown("**Parameters**")
            param_cols = [col for col in run_data.index if col.startswith('params.')]
            for col in param_cols[:5]:  # Show top 5 parameters
                param_name = col.replace('params.', '')
                st.write(f"**{param_name}:** {run_data[col]}")

        # Metrics visualization
        metric_cols = [col for col in run_data.index if col.startswith('metrics.')]
        if metric_cols:
            st.markdown("**Metrics**")

            # Create metrics chart
            metric_names = [col.replace('metrics.', '') for col in metric_cols]
            metric_values = [run_data[col] for col in metric_cols]

            # Filter numeric metrics
            numeric_metrics = []
            numeric_values = []
            for name, value in zip(metric_names, metric_values):
                try:
                    float(value)
                    numeric_metrics.append(name)
                    numeric_values.append(float(value))
                except (ValueError, TypeError):
                    continue

            if numeric_metrics:
                fig = go.Figure(data=[
                    go.Bar(x=numeric_metrics, y=numeric_values, name="Metrics")
                ])
                fig.update_layout(title="Run Metrics", xaxis_title="Metric", yaxis_title="Value")
                st.plotly_chart(fig, use_container_width=True)

    def render_performance_trends(self, runs_df: pd.DataFrame):
        """Render performance trends over time"""
        if runs_df.empty:
            return

        st.header("📈 Performance Trends")

        # Find WMAPE metric column
        wmape_cols = [col for col in runs_df.columns if 'wmape' in col.lower() and col.startswith('metrics.')]

        if not wmape_cols:
            st.warning("No WMAPE metrics found in runs.")
            return

        wmape_col = wmape_cols[0]  # Use first WMAPE column

        # Prepare data for trending
        trend_data = runs_df[runs_df[wmape_col].notna()].copy()
        trend_data = trend_data.sort_values('start_time')

        if len(trend_data) < 2:
            st.warning("Need at least 2 runs with WMAPE metrics for trending.")
            return

        # Create trend chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('WMAPE Over Time', 'Model Performance Distribution',
                          'Training Duration vs WMAPE', 'Success Rate Trend'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # WMAPE trend
        fig.add_trace(
            go.Scatter(
                x=trend_data['start_time'],
                y=trend_data[wmape_col],
                mode='lines+markers',
                name='WMAPE',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )

        # WMAPE distribution
        fig.add_trace(
            go.Histogram(
                x=trend_data[wmape_col],
                name='WMAPE Distribution',
                nbinsx=20
            ),
            row=1, col=2
        )

        # Duration vs WMAPE
        fig.add_trace(
            go.Scatter(
                x=trend_data['duration_minutes'],
                y=trend_data[wmape_col],
                mode='markers',
                name='Duration vs WMAPE',
                marker=dict(size=8, color='red')
            ),
            row=2, col=1
        )

        # Success rate over time
        trend_data['date'] = trend_data['start_time'].dt.date
        success_by_date = trend_data.groupby('date').agg({
            'status': lambda x: (x == 'FINISHED').sum() / len(x) * 100
        }).reset_index()

        fig.add_trace(
            go.Scatter(
                x=success_by_date['date'],
                y=success_by_date['status'],
                mode='lines+markers',
                name='Success Rate %',
                line=dict(color='green', width=2)
            ),
            row=2, col=2
        )

        fig.update_layout(height=800, showlegend=True, title_text="Performance Analytics")
        st.plotly_chart(fig, use_container_width=True)

    def render_business_metrics(self, runs_df: pd.DataFrame):
        """Render business-specific metrics dashboard"""
        st.header("💼 Business Metrics Dashboard")

        if runs_df.empty:
            st.warning("No runs available for business metrics analysis.")
            return

        col1, col2, col3 = st.columns(3)

        # Key Performance Indicators
        with col1:
            st.subheader("🎯 Model Accuracy KPIs")

            # Best WMAPE
            wmape_cols = [col for col in runs_df.columns if 'wmape' in col.lower() and col.startswith('metrics.')]
            if wmape_cols:
                best_wmape = runs_df[wmape_cols[0]].min()
                avg_wmape = runs_df[wmape_cols[0]].mean()

                st.metric("Best WMAPE", f"{best_wmape:.3f}")
                st.metric("Average WMAPE", f"{avg_wmape:.3f}")

                # WMAPE trend indicator
                recent_runs = runs_df.head(5)
                latest_wmape = recent_runs[wmape_cols[0]].mean()
                older_runs = runs_df.tail(5)
                older_wmape = older_runs[wmape_cols[0]].mean()

                wmape_change = latest_wmape - older_wmape
                st.metric("WMAPE Trend", f"{latest_wmape:.3f}", f"{wmape_change:.3f}")

        with col2:
            st.subheader("⚡ Operational KPIs")

            # Training efficiency
            avg_duration = runs_df['duration_minutes'].mean()
            st.metric("Avg Training Time", f"{avg_duration:.1f} min")

            # Success rate
            success_rate = (runs_df['status'] == 'FINISHED').sum() / len(runs_df) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")

            # Experiments per day
            runs_df['date'] = runs_df['start_time'].dt.date
            experiments_per_day = runs_df.groupby('date').size().mean()
            st.metric("Experiments/Day", f"{experiments_per_day:.1f}")

        with col3:
            st.subheader("📊 Resource Utilization")

            system_metrics = self.get_system_metrics()
            if system_metrics:
                st.metric("Current CPU", f"{system_metrics.get('cpu_percent', 0):.1f}%")
                st.metric("Current Memory", f"{system_metrics.get('memory_percent', 0):.1f}%")

                if 'gpu_utilization' in system_metrics:
                    st.metric("GPU Utilization", f"{system_metrics['gpu_utilization']:.1f}%")

        # Business alerts and recommendations
        self.render_business_alerts(runs_df, wmape_cols)

    def render_business_alerts(self, runs_df: pd.DataFrame, wmape_cols: List[str]):
        """Render business alerts and recommendations"""
        st.subheader("🚨 Alerts & Recommendations")

        alerts = []
        recommendations = []

        # Check WMAPE threshold
        if wmape_cols:
            latest_wmape = runs_df[wmape_cols[0]].iloc[0] if not runs_df.empty else None
            wmape_threshold = self.config.get('monitoring', {}).get('alert_thresholds', {}).get('wmape_threshold', 0.20)

            if latest_wmape and latest_wmape > wmape_threshold:
                alerts.append(f"⚠️ Latest WMAPE ({latest_wmape:.3f}) exceeds threshold ({wmape_threshold:.3f})")
                recommendations.append("Consider hyperparameter tuning or feature engineering")

        # Check success rate
        success_rate = (runs_df['status'] == 'FINISHED').sum() / len(runs_df) * 100
        if success_rate < 80:
            alerts.append(f"⚠️ Low success rate: {success_rate:.1f}%")
            recommendations.append("Investigate failing runs and improve error handling")

        # Check training time
        avg_duration = runs_df['duration_minutes'].mean()
        if avg_duration > 60:  # More than 1 hour
            alerts.append(f"⚠️ Long average training time: {avg_duration:.1f} minutes")
            recommendations.append("Consider data sampling or model optimization")

        # Display alerts
        if alerts:
            for alert in alerts:
                st.error(alert)
        else:
            st.success("✅ All systems operating within normal parameters")

        # Display recommendations
        if recommendations:
            st.markdown("**💡 Recommendations:**")
            for rec in recommendations:
                st.info(rec)

    def render_system_monitoring(self):
        """Render system monitoring section"""
        st.header("🖥️ System Monitoring")

        # Real-time system metrics
        system_metrics = self.get_system_metrics()

        if not system_metrics:
            st.error("Unable to retrieve system metrics")
            return

        col1, col2 = st.columns(2)

        with col1:
            # CPU and Memory gauges
            fig_system = make_subplots(
                rows=2, cols=2,
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}]],
                subplot_titles=("CPU Usage", "Memory Usage", "Disk Usage", "GPU Usage" if 'gpu_utilization' in system_metrics else "Network")
            )

            # CPU gauge
            fig_system.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=system_metrics.get('cpu_percent', 0),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "CPU %"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "red"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75, 'value': 90}}
                ),
                row=1, col=1
            )

            # Memory gauge
            fig_system.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=system_metrics.get('memory_percent', 0),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Memory %"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkgreen"},
                           'steps': [{'range': [0, 60], 'color': "lightgray"},
                                    {'range': [60, 85], 'color': "yellow"},
                                    {'range': [85, 100], 'color': "red"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75, 'value': 90}}
                ),
                row=1, col=2
            )

            # Disk gauge
            fig_system.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=system_metrics.get('disk_percent', 0),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Disk %"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "purple"},
                           'steps': [{'range': [0, 70], 'color': "lightgray"},
                                    {'range': [70, 90], 'color': "yellow"},
                                    {'range': [90, 100], 'color': "red"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75, 'value': 95}}
                ),
                row=2, col=1
            )

            # GPU gauge (if available)
            if 'gpu_utilization' in system_metrics:
                fig_system.add_trace(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=system_metrics.get('gpu_utilization', 0),
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "GPU %"},
                        gauge={'axis': {'range': [None, 100]},
                               'bar': {'color': "orange"},
                               'steps': [{'range': [0, 50], 'color': "lightgray"},
                                        {'range': [50, 80], 'color': "yellow"},
                                        {'range': [80, 100], 'color': "red"}],
                               'threshold': {'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75, 'value': 90}}
                    ),
                    row=2, col=2
                )

            fig_system.update_layout(height=600, font={'size': 16})
            st.plotly_chart(fig_system, use_container_width=True)

        with col2:
            # System information table
            st.subheader("System Information")

            system_info = [
                ("CPU Usage", f"{system_metrics.get('cpu_percent', 0):.1f}%"),
                ("Memory Used", f"{system_metrics.get('memory_used_gb', 0):.1f} GB"),
                ("Memory Total", f"{system_metrics.get('memory_total_gb', 0):.1f} GB"),
                ("Disk Used", f"{system_metrics.get('disk_used_gb', 0):.1f} GB"),
                ("Disk Total", f"{system_metrics.get('disk_total_gb', 0):.1f} GB"),
            ]

            if 'gpu_utilization' in system_metrics:
                system_info.extend([
                    ("GPU Utilization", f"{system_metrics.get('gpu_utilization', 0):.1f}%"),
                    ("GPU Memory", f"{system_metrics.get('gpu_memory_used', 0)} / {system_metrics.get('gpu_memory_total', 0)} MB"),
                    ("GPU Temperature", f"{system_metrics.get('gpu_temperature', 0):.1f}°C"),
                ])

            system_df = pd.DataFrame(system_info, columns=["Metric", "Value"])
            st.dataframe(system_df, use_container_width=True, hide_index=True)

    def render_sidebar(self):
        """Render dashboard sidebar"""
        with st.sidebar:
            st.header("⚙️ Dashboard Settings")

            # Auto-refresh toggle
            auto_refresh = st.checkbox("Auto Refresh", value=True)
            if auto_refresh:
                refresh_interval = st.slider("Refresh Interval (seconds)", 10, 300, self.refresh_interval)
                if refresh_interval != self.refresh_interval:
                    self.refresh_interval = refresh_interval

            st.divider()

            # MLflow connection
            st.header("🔌 MLflow Connection")
            mlflow_uri = st.text_input("MLflow URI", value=self.config.get('mlflow', {}).get('tracking_uri', 'http://localhost:5000'))

            if st.button("Reconnect MLflow"):
                self.config['mlflow']['tracking_uri'] = mlflow_uri
                self.initialize_mlflow()

            st.divider()

            # Export options
            st.header("📥 Export Options")
            if st.button("Export Metrics"):
                st.info("Metrics export functionality would be implemented here")

            if st.button("Generate Report"):
                st.info("Report generation functionality would be implemented here")

            st.divider()

            # System status
            st.header("🏥 System Status")
            status_checks = [
                ("MLflow Connection", "✅" if self.mlflow_client else "❌"),
                ("Dashboard Status", "✅"),
                ("Auto Refresh", "✅" if auto_refresh else "❌"),
            ]

            for check, status in status_checks:
                st.write(f"{check}: {status}")

    def run_dashboard(self):
        """Main dashboard execution"""
        self.render_header()

        # Initialize MLflow connection
        if not self.mlflow_client:
            if not self.initialize_mlflow():
                st.stop()

        # Render sidebar
        self.render_sidebar()

        # Main dashboard content
        try:
            # Experiment overview
            exp_data = self.render_experiment_overview()
            if exp_data:
                selected_exp, runs_df = exp_data

                # Render main sections
                self.render_runs_table(runs_df)
                self.render_performance_trends(runs_df)
                self.render_business_metrics(runs_df)

            # System monitoring
            self.render_system_monitoring()

        except Exception as e:
            st.error(f"Dashboard error: {e}")
            logger.error(f"Dashboard error: {e}")

        # Auto-refresh
        if st.sidebar.checkbox("Auto Refresh", value=False):
            time.sleep(self.refresh_interval)
            st.experimental_rerun()


def start_monitoring_dashboard(config: Optional[Dict[str, Any]] = None):
    """Start the monitoring dashboard"""
    dashboard = MLflowDashboard(config)
    dashboard.run_dashboard()


if __name__ == "__main__":
    # Run dashboard
    st.set_page_config(page_title="Forecast Monitoring", layout="wide")
    dashboard = MLflowDashboard()
    dashboard.run_dashboard()
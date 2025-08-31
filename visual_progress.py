"""
Visual Progress Tracker for GAELP Agent Training
Shows the agent getting smarter over time with visual indicators.
"""

import numpy as np
from typing import Dict, List, Optional
import time
from datetime import datetime, timedelta
from collections import deque
import math

# ANSI color codes for terminal visualization
class Colors:
    RESET = '\033[0m'
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'


class AgentEvolutionVisualizer:
    """
    Shows the agent evolving from novice to expert marketer.
    Like DeepMind's stick figures, but for marketing performance.
    """
    
    def __init__(self):
        self.evolution_stages = [
            # Stage 1: Novice (0-20% performance)
            {
                'name': 'Novice',
                'icon': '🐣',
                'description': 'Random bidding',
                'ascii_art': """
                  o
                 /|\\  <- Confused
                 / \\
                """,
                'skills': []
            },
            # Stage 2: Beginner (20-40% performance)
            {
                'name': 'Beginner',
                'icon': '🐥',
                'description': 'Learning basics',
                'ascii_art': """
                  O
                 /|\\  <- Standing
                 / \\
                """,
                'skills': ['Basic bidding']
            },
            # Stage 3: Intermediate (40-60% performance)
            {
                'name': 'Intermediate',
                'icon': '🦅',
                'description': 'Pattern recognition',
                'ascii_art': """
                  O
                </|\\> <- Walking
                 / \\
                """,
                'skills': ['Basic bidding', 'Time patterns', 'Channel preference']
            },
            # Stage 4: Advanced (60-80% performance)
            {
                'name': 'Advanced',
                'icon': '🚀',
                'description': 'Strategic planning',
                'ascii_art': """
                 \\O/
                  |   <- Running
                 / \\
                """,
                'skills': ['Basic bidding', 'Time patterns', 'Channel preference', 
                         'Competitor modeling', 'Budget pacing']
            },
            # Stage 5: Expert (80-100% performance)
            {
                'name': 'Expert',
                'icon': '🧠',
                'description': 'Superhuman performance',
                'ascii_art': """
                \\\\O//
                  |   <- Flying!
                 / \\
                """,
                'skills': ['Basic bidding', 'Time patterns', 'Channel preference',
                         'Competitor modeling', 'Budget pacing', 'Conversion prediction',
                         'Multi-touch attribution', 'Creative optimization']
            }
        ]
        
        self.current_stage = 0
        self.performance_history = deque(maxlen=100)
        self.skill_unlocks = []
        self.training_start = datetime.now()
    
    def update_performance(self, metrics: Dict) -> str:
        """Update performance and return visual representation"""
        
        # Calculate overall performance score (0-100)
        performance = self._calculate_performance_score(metrics)
        self.performance_history.append(performance)
        
        # Determine current stage
        new_stage = min(int(performance / 20), 4)
        
        # Check for stage advancement
        stage_advanced = False
        if new_stage > self.current_stage:
            stage_advanced = True
            self.current_stage = new_stage
            self._unlock_skills(new_stage)
        
        # Generate visual
        return self._generate_visual(performance, stage_advanced)
    
    def _calculate_performance_score(self, metrics: Dict) -> float:
        """Calculate overall performance score from metrics"""
        score = 0.0
        
        # Win rate (0-30 points)
        win_rate = metrics.get('win_rate', 0.0)
        score += win_rate * 30
        
        # ROI (0-30 points)
        roi = metrics.get('roi', 0.0)
        score += min(roi / 2, 1.0) * 30  # ROI of 2+ gets full points
        
        # Conversion rate (0-20 points)
        cvr = metrics.get('conversion_rate', 0.0)
        score += min(cvr / 0.02, 1.0) * 20  # 2% CVR gets full points
        
        # Learning progress (0-20 points)
        learning_rate = metrics.get('learning_progress', 0.0)
        score += learning_rate * 20
        
        return min(max(score, 0), 100)
    
    def _unlock_skills(self, stage: int):
        """Unlock new skills when advancing stages"""
        stage_info = self.evolution_stages[stage]
        for skill in stage_info['skills']:
            if skill not in self.skill_unlocks:
                self.skill_unlocks.append(skill)
    
    def _generate_visual(self, performance: float, stage_advanced: bool) -> str:
        """Generate visual representation of agent progress"""
        stage = self.evolution_stages[self.current_stage]
        
        # Build progress bar
        bar_length = 40
        filled = int(performance / 100 * bar_length)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # Color based on performance
        if performance < 20:
            color = Colors.RED
        elif performance < 40:
            color = Colors.YELLOW
        elif performance < 60:
            color = Colors.CYAN
        elif performance < 80:
            color = Colors.GREEN
        else:
            color = Colors.BRIGHT_GREEN
        
        # Build visual
        visual = f"\n{'='*60}\n"
        visual += f"{color}🎮 AGENT EVOLUTION STATUS{Colors.RESET}\n"
        visual += f"{'='*60}\n\n"
        
        # Stage and icon
        visual += f"Stage: {stage['icon']} {stage['name']} - {stage['description']}\n"
        visual += f"Performance: [{color}{bar}{Colors.RESET}] {performance:.1f}%\n\n"
        
        # ASCII art representation
        visual += f"{color}{stage['ascii_art']}{Colors.RESET}\n"
        
        # Skills unlocked
        if self.skill_unlocks:
            visual += f"\n💡 Skills Unlocked:\n"
            for skill in self.skill_unlocks[-5:]:  # Show last 5 skills
                visual += f"  ✓ {skill}\n"
        
        # Stage advancement celebration
        if stage_advanced:
            visual += f"\n{Colors.BRIGHT_YELLOW}🎉 LEVEL UP! Advanced to {stage['name']} stage!{Colors.RESET}\n"
        
        # Training time
        training_time = datetime.now() - self.training_start
        visual += f"\nTraining Time: {self._format_time(training_time)}\n"
        
        # Estimated time to expert
        if performance < 80:
            estimated_remaining = self._estimate_time_to_expert(performance)
            visual += f"Estimated to Expert: {estimated_remaining}\n"
        
        visual += f"{'='*60}\n"
        
        return visual
    
    def _format_time(self, td: timedelta) -> str:
        """Format timedelta to readable string"""
        hours = td.total_seconds() / 3600
        if hours < 1:
            return f"{int(td.total_seconds() / 60)} minutes"
        elif hours < 24:
            return f"{hours:.1f} hours"
        else:
            return f"{hours/24:.1f} days"
    
    def _estimate_time_to_expert(self, current_performance: float) -> str:
        """Estimate time remaining to reach expert level"""
        if len(self.performance_history) < 10:
            return "Calculating..."
        
        # Calculate learning rate from recent history
        recent_performance = list(self.performance_history)[-10:]
        learning_rate = (recent_performance[-1] - recent_performance[0]) / 10
        
        if learning_rate <= 0:
            return "Need more improvement"
        
        # Estimate steps to 80% performance
        remaining_performance = 80 - current_performance
        estimated_steps = remaining_performance / learning_rate
        
        # Convert to time (assuming 1 step per second)
        estimated_seconds = estimated_steps
        
        if estimated_seconds < 3600:
            return f"~{int(estimated_seconds/60)} minutes"
        elif estimated_seconds < 86400:
            return f"~{estimated_seconds/3600:.1f} hours"
        else:
            return f"~{estimated_seconds/86400:.1f} days"


class MarketingBattlefield:
    """
    Visual representation of the marketing battlefield.
    Shows agent competing against other advertisers.
    """
    
    def __init__(self):
        self.battlefield_width = 60
        self.competitors = ['Bark', 'Qustodio', 'Circle', 'Norton']
        self.our_position = 0
        self.competitor_positions = {comp: 0 for comp in self.competitors}
    
    def update_battle(self, metrics: Dict) -> str:
        """Update and visualize the marketing battlefield"""
        
        # Update positions based on performance
        our_performance = metrics.get('win_rate', 0.5)
        self.our_position = int(our_performance * self.battlefield_width)
        
        # Simulate competitor positions
        for comp in self.competitors:
            # Competitors improve slowly
            current = self.competitor_positions[comp]
            self.competitor_positions[comp] = min(
                current + np.random.randint(0, 3),
                self.battlefield_width - 5
            )
        
        # Generate battlefield visual
        visual = "\n🏁 MARKETING BATTLEFIELD\n"
        visual += "─" * (self.battlefield_width + 2) + "\n"
        
        # Our agent
        line = [' '] * self.battlefield_width
        line[min(self.our_position, self.battlefield_width-1)] = '🚀'
        visual += '│' + ''.join(line) + '│ AURA (You)\n'
        
        # Competitors
        for comp in self.competitors:
            line = [' '] * self.battlefield_width
            pos = self.competitor_positions[comp]
            line[min(pos, self.battlefield_width-1)] = '🏃'
            visual += '│' + ''.join(line) + f'│ {comp}\n'
        
        visual += "─" * (self.battlefield_width + 2) + "\n"
        visual += "Start " + " " * (self.battlefield_width - 10) + " Goal\n"
        
        # Leader board
        positions = [('AURA', self.our_position)] + \
                   [(comp, self.competitor_positions[comp]) for comp in self.competitors]
        positions.sort(key=lambda x: x[1], reverse=True)
        
        visual += "\n🏆 LEADERBOARD:\n"
        for i, (name, pos) in enumerate(positions[:3], 1):
            medal = ['🥇', '🥈', '🥉'][i-1]
            visual += f"  {medal} {name}: {pos/self.battlefield_width*100:.1f}%\n"
        
        return visual


class LearningCurveVisualizer:
    """
    Shows learning curves in ASCII art.
    Similar to TensorBoard but in terminal.
    """
    
    def __init__(self, width: int = 50, height: int = 10):
        self.width = width
        self.height = height
        self.data_points = deque(maxlen=width)
    
    def add_point(self, value: float):
        """Add a data point (0-1 range)"""
        self.data_points.append(value)
    
    def render(self, title: str = "Learning Progress") -> str:
        """Render the learning curve"""
        if len(self.data_points) < 2:
            return "Gathering data..."
        
        # Normalize data to height
        max_val = max(self.data_points) if self.data_points else 1
        min_val = min(self.data_points) if self.data_points else 0
        range_val = max_val - min_val if max_val != min_val else 1
        
        # Create grid
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Plot points
        for i, value in enumerate(self.data_points):
            normalized = (value - min_val) / range_val
            y = int((1 - normalized) * (self.height - 1))
            y = max(0, min(self.height - 1, y))
            
            # Use different characters for trend
            if i > 0:
                prev_value = self.data_points[i-1]
                if value > prev_value:
                    char = '/'
                elif value < prev_value:
                    char = '\\'
                else:
                    char = '-'
            else:
                char = '•'
            
            grid[y][i] = char
        
        # Build visual
        visual = f"\n📊 {title}\n"
        visual += "┌" + "─" * self.width + "┐\n"
        
        for row in grid:
            visual += "│" + ''.join(row) + "│\n"
        
        visual += "└" + "─" * self.width + "┘\n"
        visual += f"  {min_val:.2f}" + " " * (self.width - 10) + f"{max_val:.2f}\n"
        
        # Add trend indicator
        if len(self.data_points) >= 10:
            recent_avg = sum(list(self.data_points)[-5:]) / 5
            older_avg = sum(list(self.data_points)[-10:-5]) / 5
            
            if recent_avg > older_avg * 1.1:
                trend = "📈 Improving rapidly!"
            elif recent_avg > older_avg:
                trend = "📈 Improving"
            elif recent_avg < older_avg * 0.9:
                trend = "📉 Declining"
            else:
                trend = "➡️ Stable"
            
            visual += f"Trend: {trend}\n"
        
        return visual


class ComprehensiveProgressTracker:
    """
    Combines all visualizations into a comprehensive progress display.
    """
    
    def __init__(self):
        self.evolution = AgentEvolutionVisualizer()
        self.battlefield = MarketingBattlefield()
        self.roi_curve = LearningCurveVisualizer(width=40, height=8)
        self.win_rate_curve = LearningCurveVisualizer(width=40, height=8)
        
        self.update_count = 0
        self.last_display_time = time.time()
    
    def update_and_display(self, metrics: Dict, force_display: bool = False):
        """Update all visualizations and display if appropriate"""
        
        self.update_count += 1
        
        # Update curves
        self.roi_curve.add_point(min(metrics.get('roi', 0) / 3, 1))  # Normalize ROI to 0-1
        self.win_rate_curve.add_point(metrics.get('win_rate', 0))
        
        # Display every 10 updates or every 5 seconds
        current_time = time.time()
        should_display = (
            force_display or 
            self.update_count % 10 == 0 or 
            current_time - self.last_display_time > 5
        )
        
        if should_display:
            # Clear screen (optional - comment out if you want history)
            # print("\033[2J\033[H")
            
            # Display evolution
            print(self.evolution.update_performance(metrics))
            
            # Display battlefield
            print(self.battlefield.update_battle(metrics))
            
            # Display learning curves side by side
            roi_lines = self.roi_curve.render("ROI Progress").split('\n')
            win_lines = self.win_rate_curve.render("Win Rate").split('\n')
            
            print("\n📈 LEARNING CURVES")
            print("="*85)
            for roi_line, win_line in zip(roi_lines, win_lines):
                print(f"{roi_line:<42} {win_line}")
            
            # Summary stats
            print("\n📊 QUICK STATS")
            print("="*85)
            print(f"Episodes: {metrics.get('episodes', 0):,} | "
                  f"Total Spend: ${metrics.get('total_spend', 0):,.2f} | "
                  f"Total Revenue: ${metrics.get('total_revenue', 0):,.2f}")
            print(f"Best ROI: {metrics.get('best_roi', 0):.2f} | "
                  f"Current ROI: {metrics.get('roi', 0):.2f} | "
                  f"Conversions: {metrics.get('conversions', 0):,}")
            
            self.last_display_time = current_time
    
    def get_training_duration_estimate(self) -> str:
        """Estimate total training duration needed"""
        
        # For Aura Balance's complexity:
        # - 100,000 episodes for basic competence
        # - 500,000 episodes for advanced strategies
        # - 1,000,000+ episodes for superhuman performance
        
        # At 1000 episodes/second (with optimizations):
        basic_hours = 100000 / 1000 / 3600
        advanced_hours = 500000 / 1000 / 3600
        superhuman_hours = 1000000 / 1000 / 3600
        
        estimate = f"""
╔══════════════════════════════════════════════════════════════╗
║               TRAINING DURATION ESTIMATES                     ║
╠══════════════════════════════════════════════════════════════╣
║ 🐣 Basic Competence:    ~{basic_hours:.1f} hours                       ║
║    - Learn bidding basics                                    ║
║    - Understand time patterns                                ║
║    - Channel preferences                                     ║
║                                                              ║
║ 🦅 Advanced Strategies: ~{advanced_hours:.1f} hours                       ║
║    - Competitor modeling                                     ║
║    - Multi-touch attribution                                 ║
║    - Budget optimization                                     ║
║                                                              ║
║ 🧠 Superhuman Level:    ~{superhuman_hours:.1f} hours                       ║
║    - Discover non-obvious patterns                          ║
║    - Perfect timing and creative selection                  ║
║    - Outperform human marketers                            ║
╠══════════════════════════════════════════════════════════════╣
║ 💡 Recommendation: Run overnight for basic competence        ║
║                   Run 2-3 days for production ready         ║
╚══════════════════════════════════════════════════════════════╝
        """
        return estimate


# Demo function
def demo_visual_progress():
    """Demonstrate the visual progress tracking"""
    
    tracker = ComprehensiveProgressTracker()
    
    print(tracker.get_training_duration_estimate())
    time.sleep(2)
    
    # Simulate training progress
    for episode in range(100):
        # Simulate improving metrics
        metrics = {
            'episodes': episode * 100,
            'win_rate': min(0.1 + episode * 0.008, 0.85),
            'roi': 0.5 + episode * 0.025,
            'conversion_rate': 0.005 + episode * 0.0002,
            'learning_progress': min(episode / 100, 1.0),
            'total_spend': episode * 100,
            'total_revenue': episode * 150,
            'conversions': episode * 2,
            'best_roi': 0.5 + episode * 0.03
        }
        
        tracker.update_and_display(metrics)
        time.sleep(0.5)  # Simulate training time


if __name__ == "__main__":
    demo_visual_progress()
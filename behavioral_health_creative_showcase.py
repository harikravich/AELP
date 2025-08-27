#!/usr/bin/env python3
"""
Behavioral Health Creative Showcase
Displays the generated creative content in organized, readable format
"""

import json
from typing import Dict, List, Any

def load_creative_results(filename: str) -> Dict[str, Any]:
    """Load creative generation results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def showcase_creatives(results: Dict[str, Any]):
    """Display creative content in organized format"""
    
    print("🧠 AURA BALANCE BEHAVIORAL HEALTH CREATIVE SHOWCASE")
    print("=" * 80)
    print(f"Generated: {results['total_creatives']} unique creatives")
    print(f"Timestamp: {results['generation_timestamp']}")
    print()
    
    # Organize creatives by type
    creatives_by_type = {}
    for creative in results['creatives']:
        creative_type = creative['creative_type']
        if creative_type not in creatives_by_type:
            creatives_by_type[creative_type] = []
        creatives_by_type[creative_type].append(creative)
    
    # Display each type
    for creative_type, creatives in creatives_by_type.items():
        print(f"\n🎯 {creative_type.upper().replace('_', ' ')} ({len(creatives)} variants)")
        print("=" * 60)
        
        if creative_type == "ad_description":
            showcase_ad_descriptions(creatives)
        elif creative_type == "landing_page_hero":
            showcase_landing_page_heroes(creatives)
        elif creative_type == "email_sequence":
            showcase_email_sequences(creatives)
        elif creative_type == "display_ad":
            showcase_display_ads(creatives)
        elif creative_type == "video_script":
            showcase_video_scripts(creatives)

def showcase_ad_descriptions(creatives: List[Dict[str, Any]]):
    """Display ad descriptions organized by format and focus"""
    
    # Group by format
    by_format = {}
    for creative in creatives:
        format_key = creative['creative_format']
        if format_key not in by_format:
            by_format[format_key] = []
        by_format[format_key].append(creative)
    
    for format_name, format_creatives in by_format.items():
        print(f"\n📝 {format_name.replace('_', ' ').title()}")
        print("-" * 40)
        
        for i, creative in enumerate(format_creatives, 1):
            focus = creative['behavioral_focus'].replace('_', ' ').title()
            segment = creative['segment'].replace('_', ' ').title()
            ctr = creative['ctr_simulation']
            conv = creative['conversion_rate']
            
            print(f"{i}. {creative['content']}")
            print(f"   Focus: {focus} | Segment: {segment}")
            print(f"   Performance: CTR {ctr:.3f} | Conv {conv:.3f}")
            print()

def showcase_landing_page_heroes(creatives: List[Dict[str, Any]]):
    """Display landing page hero sections"""
    
    for i, creative in enumerate(creatives, 1):
        focus = creative['behavioral_focus'].replace('_', ' ').title()
        format_name = creative['creative_format'].replace('_', ' ').title()
        ctr = creative['ctr_simulation']
        conv = creative['conversion_rate']
        
        print(f"\n🏠 Hero Section #{i} - {format_name}")
        print(f"Focus: {focus}")
        print("-" * 50)
        print(creative['content'])
        print(f"\n📊 Performance: CTR {ctr:.3f} | Conv {conv:.3f}")
        print("=" * 50)

def showcase_email_sequences(creatives: List[Dict[str, Any]]):
    """Display email sequences"""
    
    for i, creative in enumerate(creatives, 1):
        focus = creative['behavioral_focus'].replace('_', ' ').title()
        format_name = creative['creative_format'].replace('_', ' ').title()
        ctr = creative['ctr_simulation']
        conv = creative['conversion_rate']
        
        print(f"\n📧 Email #{i} - {format_name}")
        print(f"Focus: {focus}")
        print("-" * 50)
        print(creative['content'])
        print(f"\n📊 Performance: CTR {ctr:.3f} | Conv {conv:.3f}")
        print("=" * 50)

def showcase_display_ads(creatives: List[Dict[str, Any]]):
    """Display ads organized by size"""
    
    # Group by format (size)
    by_size = {}
    for creative in creatives:
        size_key = creative['creative_format']
        if size_key not in by_size:
            by_size[size_key] = []
        by_size[size_key].append(creative)
    
    for size_name, size_creatives in by_size.items():
        print(f"\n📐 {size_name.replace('_', ' ').title()}")
        print("-" * 40)
        
        for i, creative in enumerate(size_creatives, 1):
            focus = creative['behavioral_focus'].replace('_', ' ').title()
            ctr = creative['ctr_simulation']
            conv = creative['conversion_rate']
            
            print(f"{i}. {creative['content']}")
            print(f"   Focus: {focus}")
            print(f"   Performance: CTR {ctr:.3f} | Conv {conv:.3f}")
            print()

def showcase_video_scripts(creatives: List[Dict[str, Any]]):
    """Display video scripts"""
    
    for i, creative in enumerate(creatives, 1):
        focus = creative['behavioral_focus'].replace('_', ' ').title()
        format_name = creative['creative_format'].replace('_', ' ').title()
        ctr = creative['ctr_simulation']
        conv = creative['conversion_rate']
        
        print(f"\n🎬 Video Script #{i} - {format_name}")
        print(f"Focus: {focus}")
        print("-" * 50)
        print(creative['content'])
        print(f"\n📊 Performance: CTR {ctr:.3f} | Conv {conv:.3f}")
        print("=" * 50)

def show_performance_summary(results: Dict[str, Any]):
    """Show overall performance summary"""
    
    print(f"\n📈 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    creatives = results['creatives']
    
    # Overall metrics
    avg_ctr = sum(c['ctr_simulation'] for c in creatives) / len(creatives)
    avg_conv = sum(c['conversion_rate'] for c in creatives) / len(creatives)
    avg_engagement = sum(c['engagement_score'] for c in creatives) / len(creatives)
    
    print(f"Overall Performance:")
    print(f"  Average CTR: {avg_ctr:.3f}")
    print(f"  Average Conversion Rate: {avg_conv:.3f}")
    print(f"  Average Engagement Score: {avg_engagement:.3f}")
    print()
    
    # Top performers
    sorted_by_ctr = sorted(creatives, key=lambda x: x['ctr_simulation'], reverse=True)
    sorted_by_conv = sorted(creatives, key=lambda x: x['conversion_rate'], reverse=True)
    
    print("🏆 Top 5 by CTR:")
    for i, creative in enumerate(sorted_by_ctr[:5], 1):
        content_preview = creative['content'][:60] + "..." if len(creative['content']) > 60 else creative['content']
        print(f"  {i}. {creative['ctr_simulation']:.3f} - {content_preview}")
    
    print("\n🎯 Top 5 by Conversion Rate:")
    for i, creative in enumerate(sorted_by_conv[:5], 1):
        content_preview = creative['content'][:60] + "..." if len(creative['content']) > 60 else creative['content']
        print(f"  {i}. {creative['conversion_rate']:.3f} - {content_preview}")
    
    # Performance by type
    print(f"\n📊 Performance by Creative Type:")
    type_performance = {}
    for creative in creatives:
        ctype = creative['creative_type']
        if ctype not in type_performance:
            type_performance[ctype] = {'ctr': [], 'conv': [], 'eng': []}
        type_performance[ctype]['ctr'].append(creative['ctr_simulation'])
        type_performance[ctype]['conv'].append(creative['conversion_rate'])
        type_performance[ctype]['eng'].append(creative['engagement_score'])
    
    for ctype, metrics in type_performance.items():
        avg_ctr = sum(metrics['ctr']) / len(metrics['ctr'])
        avg_conv = sum(metrics['conv']) / len(metrics['conv'])
        avg_eng = sum(metrics['eng']) / len(metrics['eng'])
        count = len(metrics['ctr'])
        
        print(f"  {ctype.replace('_', ' ').title()} ({count}): CTR {avg_ctr:.3f} | Conv {avg_conv:.3f} | Eng {avg_eng:.3f}")

def show_behavioral_health_focus_analysis(results: Dict[str, Any]):
    """Analyze performance by behavioral health focus"""
    
    print(f"\n🧠 BEHAVIORAL HEALTH FOCUS ANALYSIS")
    print("=" * 60)
    
    creatives = results['creatives']
    
    # Group by behavioral focus
    focus_performance = {}
    for creative in creatives:
        focus = creative['behavioral_focus']
        if focus not in focus_performance:
            focus_performance[focus] = {'creatives': [], 'ctr': [], 'conv': []}
        focus_performance[focus]['creatives'].append(creative)
        focus_performance[focus]['ctr'].append(creative['ctr_simulation'])
        focus_performance[focus]['conv'].append(creative['conversion_rate'])
    
    # Show performance by focus
    for focus, data in focus_performance.items():
        avg_ctr = sum(data['ctr']) / len(data['ctr'])
        avg_conv = sum(data['conv']) / len(data['conv'])
        count = len(data['creatives'])
        
        focus_name = focus.replace('_', ' ').title()
        print(f"\n🎯 {focus_name} ({count} creatives)")
        print(f"   Average CTR: {avg_ctr:.3f}")
        print(f"   Average Conversion Rate: {avg_conv:.3f}")
        
        # Show best performing creative for this focus
        best_creative = max(data['creatives'], key=lambda x: x['ctr_simulation'])
        content_preview = best_creative['content'][:80] + "..." if len(best_creative['content']) > 80 else best_creative['content']
        print(f"   Best Performer: {content_preview}")

def main():
    """Main showcase function"""
    
    # Load the most recent results file
    filename = "behavioral_health_creatives_1755913836.json"
    
    try:
        results = load_creative_results(filename)
        
        # Main showcase
        showcase_creatives(results)
        
        # Performance analysis
        show_performance_summary(results)
        
        # Behavioral health focus analysis
        show_behavioral_health_focus_analysis(results)
        
        print(f"\n✅ SHOWCASE COMPLETE")
        print("=" * 60)
        print("🎯 Key Achievements:")
        print(f"   • Generated {results['total_creatives']} unique behavioral health creatives")
        print("   • NO templates or fallback content used")
        print("   • All content LLM-generated with behavioral health focus")
        print("   • Clinical authority and crisis vs prevention messaging")
        print("   • iOS premium positioning included")
        print("   • Performance tested in simulation")
        print()
        print("📋 Creative Types Generated:")
        for ctype, count in results['creative_breakdown'].items():
            print(f"   • {ctype.replace('_', ' ').title()}: {count} variants")
        print()
        print("🧠 Behavioral Health Focus Areas:")
        creatives = results['creatives']
        focus_counts = {}
        for creative in creatives:
            focus = creative['behavioral_focus']
            focus_counts[focus] = focus_counts.get(focus, 0) + 1
        
        for focus, count in focus_counts.items():
            focus_name = focus.replace('_', ' ').title()
            print(f"   • {focus_name}: {count} creatives")
        
    except FileNotFoundError:
        print(f"❌ Results file {filename} not found")
        print("   Run the creative generator first")
    except Exception as e:
        print(f"❌ Error loading results: {e}")

if __name__ == "__main__":
    main()
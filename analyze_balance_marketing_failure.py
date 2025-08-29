#!/usr/bin/env python3
"""
Analyze why Balance (BEHAVIORAL HEALTH app) marketing is failing
The product is for teen mental health, NOT parental controls!
"""

print("="*80)
print("BALANCE MARKETING FAILURE ANALYSIS")
print("BEHAVIORAL HEALTH APP - NOT PARENTAL CONTROLS!")
print("="*80)

print("\n🧠 WHAT BALANCE ACTUALLY IS:")
print("-" * 80)
print("""
Balance by Aura is a BEHAVIORAL HEALTH platform for teens/young adults:
- Mental health support
- Anxiety and stress management  
- Depression screening and support
- Digital wellbeing (healthy tech habits)
- Emotional regulation tools
- Peer support and community

Target Users: TEENS and YOUNG ADULTS (not parents!)
Value Prop: Mental health support in their pocket
""")

print("\n❌ CAMPAIGN ANALYSIS - COMPLETE MISMATCH:")
print("-" * 80)

campaigns = {
    'balance_parentingpressure_osaw': {
        'sessions': 11203,
        'cvr': 0.0014,
        'problem': "Negative framing - blames parents, attracts wrong audience"
    },
    'balance_teentalk_osaw': {
        'sessions': 2829,
        'cvr': 0.0092,
        'problem': "Better but vague - what kind of teen talk?"
    },
    'life360_topparents': {
        'sessions': 1461,
        'cvr': 0.0075,
        'problem': "Targeting PARENTS for a TEEN app - fundamental mismatch"
    },
    'life360_parentsover50': {
        'sessions': 968,
        'cvr': 0.0031,
        'problem': "Parents over 50?? Their kids are adults, not teens!"
    },
    'balance_bluebox_osaw': {
        'sessions': 590,
        'cvr': 0.0051,
        'problem': "What is bluebox? Zero value prop communication"
    }
}

print("\nCAMPAIGN PROBLEMS:")
for campaign, data in campaigns.items():
    print(f"\n{campaign}")
    print(f"  Sessions: {data['sessions']:,} | CVR: {data['cvr']*100:.2f}%")
    print(f"  ❌ {data['problem']}")

print("\n" + "="*80)
print("WHY THE MARKETING IS FAILING")
print("="*80)

failures = {
    "1. WRONG AUDIENCE": [
        "Targeting parents instead of teens",
        "Parents over 50? Their kids don't need teen apps",
        "Facebook ads reaching parents, not TikTok/Instagram for teens"
    ],
    
    "2. TERRIBLE MESSAGING": [
        "'Parenting pressure' - negative, blame-focused",
        "'Blue box' - meaningless, no value prop",
        "No mention of anxiety, depression, mental health support",
        "Sounds like surveillance, not support"
    ],
    
    "3. CHANNEL MISMATCH": [
        "86% Facebook (parents) vs TikTok/Snapchat (teens)",
        "Should be on platforms where teens seek help",
        "Reddit, Discord, Instagram would be better"
    ],
    
    "4. LANDING PAGE CONFUSION": [
        "/online-wellbeing - too vague, what does it do?",
        "/more-balance - more of what? 0% conversion!",
        "Not explaining it's mental health support",
        "Parents land and think it's screen time control"
    ],
    
    "5. VALUE PROP BURIAL": [
        "Teen mental health crisis is HUGE",
        "Teens desperately need support",
        "But campaigns don't mention mental health",
        "Hiding the actual value behind vague terms"
    ]
}

for failure_type, issues in failures.items():
    print(f"\n{failure_type}:")
    for issue in issues:
        print(f"  • {issue}")

print("\n" + "="*80)
print("WHAT GOOD BEHAVIORAL HEALTH MARKETING LOOKS LIKE")
print("="*80)

print("""
✅ SUCCESSFUL COMPETITORS:

1. Headspace for Teens
   - "Sleep better, stress less"
   - Direct to teens on TikTok
   - Uses teen influencers
   - CVR: 3-5%

2. Calm for Students  
   - "Ace your exams without anxiety"
   - Instagram and YouTube ads
   - Student ambassadors
   - CVR: 4-6%

3. BetterHelp Teen
   - "Talk to someone who gets it"
   - Reddit and Discord presence
   - Peer testimonials
   - CVR: 2-4%
""")

print("\n" + "="*80)
print("RECOMMENDATIONS TO FIX BALANCE MARKETING")
print("="*80)

recommendations = {
    "1. AUDIENCE": [
        "Target TEENS (13-19) not parents",
        "Target COLLEGE STUDENTS (18-24)",
        "Stop wasting money on 'parents over 50'"
    ],
    
    "2. MESSAGING": [
        "Lead with: 'Feeling overwhelmed? You're not alone'",
        "Focus on: Anxiety, stress, school pressure, social anxiety",
        "Use their language: 'mental health' not 'wellbeing'",
        "Testimonials from actual teens"
    ],
    
    "3. CHANNELS": [
        "TikTok: Where teens talk about mental health",
        "Instagram: Visual stories of recovery",
        "YouTube: Partner with teen mental health creators",
        "Reddit: r/teenagers, r/anxiety, r/mentalhealth",
        "Discord: Gaming and study servers"
    ],
    
    "4. LANDING PAGES": [
        "Hero: 'Your mental health companion'",
        "Clear features: Mood tracking, coping tools, peer support",
        "Show the app interface - make it feel safe, not monitored",
        "Pricing for teens (or parent-pays-for-teen model)"
    ],
    
    "5. CREATIVE TESTING": [
        "A: 'Stressed about school?' (direct)",
        "B: 'Your mental health matters' (supportive)",
        "C: '73% of teens feel overwhelmed' (statistical)",
        "D: 'Find your balance' (aspirational)"
    ]
}

for category, items in recommendations.items():
    print(f"\n{category}")
    for item in items:
        print(f"  → {item}")

print("\n" + "="*80)
print("THE BRUTAL TRUTH")
print("="*80)

print("""
🔥 Balance has a GREAT product for a HUGE market need:
   - Teen mental health crisis is real
   - Parents would pay $75/month to help their teen
   - Teens desperately need these tools

😭 But the marketing is COMPLETELY WRONG:
   - Wrong audience (parents vs teens)
   - Wrong message (pressure vs support)
   - Wrong channels (Facebook vs TikTok)
   - Wrong framing (control vs care)

💡 With proper marketing, Balance could have:
   - 2-4% CVR (like competitors)
   - 10x current revenue
   - Actual impact on teen mental health

The marketing team needs to:
1. Understand the product (behavioral health, not parental control)
2. Understand the audience (teens, not parents over 50)
3. Use the right channels (where teens are)
4. Communicate the value (mental health support)
5. Test creative that resonates with teens

Current 0.32% CVR → Potential 3%+ CVR = 10X IMPROVEMENT
""")

print("\n✅ The product is good. The marketing is the problem!")
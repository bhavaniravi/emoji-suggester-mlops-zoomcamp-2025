import pandas as pd
import random

# Emoticons and their corresponding labels
emoticons = [
    "😭",
    "😡",
    "😱",
    "😴",
    "😇",
    "🤔",
    "🤯",
    "😤",
    "😈",
    "👀",
    "🥳",
    "🤗",
    "😅",
    "😬",
    "🤩",
    "😷",
    "😵",
    "🤒",
    "🤢",
    "🤮",
]
labels = list(range(20, 40))

# Mock tweet templates
emotion_templates = {
    20: [
        "I’ve been crying all night after that.",
        "Tears won't stop coming.",
        "That broke my heart completely.",
    ],
    21: [
        "Can’t believe how angry I still am.",
        "I'm furious and done with this.",
        "Why am I this mad?",
    ],
    22: [
        "That jump scare nearly killed me.",
        "I was genuinely terrified.",
        "Still shaking from that horror clip.",
    ],
    23: [
        "I’m heading to bed. So exhausted.",
        "Can't keep my eyes open.",
        "Falling asleep while standing.",
    ],
    24: [
        "Trying to stay pure and kind.",
        "Just doing the right thing today.",
        "Helping others makes me feel good.",
    ],
    25: [
        "Not sure what to think right now.",
        "Pondering life’s questions again.",
        "I need time to think this through.",
    ],
    26: [
        "Mind blown by that plot twist.",
        "What just happened?!",
        "Still processing what I saw.",
    ],
    27: [
        "Ugh, that was so frustrating.",
        "Nothing is going right today.",
        "Sick of repeating myself.",
    ],
    28: [
        "Feeling a little mischievous today.",
        "Time to stir some trouble.",
        "Why follow rules when I can bend them?",
    ],
    29: [
        "I swear someone’s watching me.",
        "Eyes on me everywhere I go.",
        "Being observed constantly.",
    ],
    30: [
        "This party is going to be wild!",
        "Let’s celebrate all night long!",
        "Ready to dance till sunrise.",
    ],
    31: [
        "Sending virtual hugs to everyone.",
        "Need a hug right now.",
        "Big hugs make everything better.",
    ],
    32: [
        "That was awkward...",
        "Oops, didn’t mean to do that.",
        "Why am I always this clumsy?",
    ],
    33: [
        "Embarrassing moment of the week.",
        "I cringed so hard at myself.",
        "Just pretend that didn’t happen.",
    ],
    34: [
        "This is absolutely amazing!",
        "I'm so impressed right now.",
        "Best day ever, hands down!",
    ],
    35: [
        "Feeling under the weather today.",
        "Wearing a mask and staying in.",
        "Coughing nonstop since morning.",
    ],
    36: [
        "Everything is spinning. Need to lie down.",
        "Feeling dizzy all day.",
        "Can't focus. Head is spinning.",
    ],
    37: [
        "Woke up with a fever. Ugh.",
        "Sick and staying in bed.",
        "Need soup and sleep immediately.",
    ],
    38: [
        "My stomach is upset again.",
        "Food poisoning is no joke.",
        "Regretting that sketchy lunch.",
    ],
    39: [
        "That was disgusting. I feel sick.",
        "Shouldn’t have eaten that.",
        "I’m about to throw up.",
    ],
}

# Generate 100 rows of text samples
records = []
for _ in range(100):
    label = random.choice(labels)
    text = random.choice(emotion_templates[label])
    records.append({"text": text, "label": label})

# Save to CSV files
tweets_df = pd.DataFrame(records)
mapping_df = pd.DataFrame({"emoticon": emoticons, "number": labels})

tweets_df.to_csv("data/mock/test.csv", index=False)
mapping_df.to_csv("data/mock/mapping.csv", index=False)

print("Files 'text.csv' and 'mapping.csv' have been created.")

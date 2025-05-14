import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Küçük ve hızlı embedding modeli
model = SentenceTransformer('all-MiniLM-L6-v2')

# Kategori seçenekleri
content_options = [
    "science", "technology", "sports", "food", "fashion",
    "entertainment", "parenting", "politics", "animals", "academic", "economy"
]

# Açıklamalar
content_descriptions = {
    "science": (
        "Scientific domains including research, space missions (e.g., NASA, SpaceX), "
        "physics, chemistry, biology news (e.g., DNA, cells), scientific institutions, or universities."
    ),
    "technology": (
        "Developments in artificial intelligence (e.g., ChatGPT), software engineering, "
        "cybersecurity, startup innovations, gadgets, hardware, and IT trends."
    ),
    "sports": (
        "This article covers sports competitions, professional games such as football, basketball, tennis, athletics, and includes topics like athletes, player stats,team performance, match scores, league results, championships, tournaments, Olympic events, and sports news coverage."
    ),
    "food": (
        "This article is about culinary topics such as recipes, cooking techniques, nutrition advice, trending diets, restaurant reviews, "
        "traditional cuisine, food culture, healthy eating, chef interviews, or gourmet experiences."
    ),

    "fashion": (
        "This article focuses on the fashion and beauty industry, including clothing trends, fashion week events, famous designers, personal styling, "
        "runway shows, seasonal collections, makeup tutorials, skincare tips, influencer fashion, and beauty products."
    ),

    "entertainment": (
        "This article covers the entertainment industry including movies, television shows, music releases, concerts, celebrity interviews, award shows like the Oscars or Grammys, "
        "streaming platforms such as Netflix or Hulu, viral trends, and pop culture events."
    ),

    "parenting": (
        "This article discusses family-related content including parenting strategies, raising children, emotional bonding, child psychology, "
        "education and schooling, managing family life, household activities, parental challenges, and child development milestones."

),
"politics": (
    "This article relates to political topics including domestic or international policy, elections, "
    "trade negotiations, diplomatic relations between countries, government decisions, political agreements, "
    "and international summits involving political figures or institutions."
)
,
"animals": (
        "This article involves animals, including pets like dogs and cats, wildlife species, veterinary medicine, zoos and aquariums, endangered animals, "
        "animal rights and welfare, rescue stories, conservation efforts, and human-animal interactions."
    ),

    "academic": (
        "This article is related to education, academia, and learning — including universities, schools, research institutions, academic journals, "
        "teaching methods, exam preparation, student experiences, higher education policies, and scholarly research."
    ),

    "economy": (
        "This article covers economic topics such as global finance, inflation rates, stock market trends, employment and job statistics, government spending, "
        "GDP growth, interest rates, economic crises, business outlooks, and international trade or monetary policy."
    )
}
# Tahmin fonksiyonu
def predict(text, selected_category):
    if selected_category not in content_descriptions:
        return "⚠️ Invalid category selected."

    text_embedding = model.encode([text])[0]
    category_embedding = model.encode([content_descriptions[selected_category]])[0]
    similarity = cosine_similarity([text_embedding], [category_embedding])[0][0]

    result = "✅ Relevant" if similarity >= 0.24 else "❌ Non-Relevant"
    return f"{result} (Similarity Score: {similarity:.2f})"

# Arayüz
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Enter your text", lines=4, placeholder="Paste your text here..."),
        gr.Dropdown(choices=content_options, label="Select content category")
    ],
    outputs="text",
    title="Content Relevance Checker (Semantic Match)",
    description="Paste a news text and select a topic category. It will check semantic relevance using cosine similarity."
)

interface.launch(share=True)



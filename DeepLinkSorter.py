import json, torch, os, requests, re, faiss, numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


class SubredditProcessor:
    def __init__(self, model_name='BAAI/bge-base-en-v1.5', filepath='DeepLinkSorter_index'):
        self.model = self.load_model(model_name)
        self.filepath = filepath
    
    @staticmethod
    def load_model(model_name):
        return SentenceTransformer(model_name)
    
    @staticmethod
    def save_model(model, filepath):
        model.save(filepath)

    def load_embeddings_and_index(self):
        index = faiss.read_index(f"{self.filepath}.index")
        link_embeddings = np.load(f"{self.filepath}_embeddings.npy")
        return torch.tensor(link_embeddings), index
    
    def update_embeddings_and_index(self, new_embeddings: torch.Tensor):
        embeddings_file = f"{self.filepath}_embeddings.npy"
        existing_embeddings = np.load(embeddings_file) if os.path.exists(embeddings_file) else np.empty((0, new_embeddings.shape[1]))
        updated_embeddings = np.vstack((existing_embeddings, new_embeddings.cpu().detach().numpy()))
        np.save(embeddings_file, updated_embeddings)
        
        index = faiss.read_index(f"{self.filepath}.index") if os.path.exists(f"{self.filepath}.index") else faiss.IndexFlatIP(new_embeddings.shape[1])
        index.add(new_embeddings.cpu().detach().numpy())
        faiss.write_index(index, f"{self.filepath}.index")
    
    @staticmethod
    def process_comments(comments_link):
        headers = {'User-Agent': 'Mozilla/5.0'}
        comments = []
        if comments_link:
            response = requests.get(comments_link, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            comment_elements = soup.find_all('div', class_='md')
            for comment in comment_elements:
                clean_text = SubredditProcessor.clean_comment(comment.get_text())
                comments.append(clean_text)
        return comments
    
    @staticmethod
    def clean_comment(comment: str) -> str:
        patterns_to_remove = [
            r"Rules For Posts.*?Related Subreddit :",
            r"AMAs:.*?(?=Related Subreddit :|$)",
            r"Advanced Courses.*?(?=AMAs:|$)",
            r"Beginners:.*?(?=Advanced Courses:|$)",
            r"@[\w]+",  # Handles Twitter handles
            r"/r/[\w]+",  # Handles subreddit references
        ]
        for pattern in patterns_to_remove:
            comment = re.sub(pattern, '', comment, flags=re.DOTALL)
        comment = re.sub(r'\s+', ' ', comment).strip()
        comment = comment.replace(u'\u2019', "'").replace(u'\u00fc', 'ue').replace('\\', '')
        return comment
    


class PostDataset:
    def __init__(self, filepath='processed_posts.json'):
        self.filepath = filepath
        self.posts_set = self.load_processed_posts_set()

    def load_processed_posts_set(self) -> set:
        try:
            with open(self.filepath, 'r') as file:
                return set(json.load(file))
        except FileNotFoundError:
            return set()

    def update_processed_posts_set(self, new_posts_data: list):
        new_links = {post['comments_link'] for post in new_posts_data}
        self.posts_set.update(new_links)
        with open(self.filepath, 'w') as file:
            json.dump(list(self.posts_set), file)

    def check_if_post_processed(self, comments_link):
        return comments_link in self.posts_set


class SubredditScraper:
    def __init__(self, user_agent='Mozilla/5.0'):
        self.headers = {'User-Agent': user_agent}

    def scrape_popular(self, limit=50):
        base_url = 'https://old.reddit.com/r/popular/'
        return self.scrape_subreddit_base(base_url, limit)

    def scrape_subreddit_base(self, base_url, limit=50):
        posts_data = []
        current_count = 0
        url = base_url
        while current_count < limit:
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                print(f"Error fetching the subreddit: {response.status_code}")
                break

            soup = BeautifulSoup(response.text, 'html.parser')
            posts = soup.find_all('div', class_='thing')

            for post in posts:
                if current_count < limit:
                    title = post.find('a', class_='title').text
                    comments_element = post.find('a', class_='comments')
                    comments_link = comments_element.get('href') if comments_element else None
                    if comments_link:
                        posts_data.append({'title': title, 'comments_link': comments_link})
                        current_count += 1
                else:
                    break

            next_button = soup.find('span', class_='next-button')
            if next_button:
                next_page_link = next_button.find('a').get('href')
                url = next_page_link
            else:
                break
        return posts_data


class LinkSorter:
    def __init__(self, model: SentenceTransformer, dataset: PostDataset, scraper: SubredditScraper):
        self.model = model
        self.dataset = dataset
        self.scraper = scraper


    def process_popular_subreddits(self, limit_per_subreddit=10):
        print("Starting process for popular subreddits")
        
        popular_posts_data = self.scraper.scrape_popular(limit=limit_per_subreddit)
        print(f"Retrieved {len(popular_posts_data)} posts from popular subreddits")

        new_posts_data = [post for post in popular_posts_data if not self.dataset.check_if_post_processed(post['comments_link'])]
        print(f"Found {len(new_posts_data)} new posts to process from popular subreddits")

        if new_posts_data:
            sorted_posts, post_embeddings, index = self.sort_links(new_posts_data, ['deep learning', 'neural network', 'AI']) #These are the goal keywords you are searching for
            
            self.save_embeddings_and_index(post_embeddings, index)
            self.dataset.update_processed_posts_set(new_posts_data)
            print("Processed and saved data for popular subreddits")
        else:
            print("No new posts to process or all posts have been previously processed from popular subreddits")

    def save_embeddings_and_index(self, link_embeddings: torch.Tensor, index: faiss.IndexFlatIP, filepath='popular_subreddits'):
        if not isinstance(link_embeddings, list) or len(link_embeddings) > 0:
            faiss.write_index(index, f"{filepath}.index")
            np.save(f"{filepath}_embeddings.npy", link_embeddings.cpu().detach().numpy())
        else:
            print("No embeddings to save. The list of link embeddings is empty.")

    def sort_links(self, posts: list, goal_keywords: list, GPU: int = None) -> tuple:
        for post in posts:
            post['comments'] = SubredditProcessor.process_comments(post['comments_link'])
        titles_and_comments = [" ".join([post['title']] + post['comments']) for post in posts]

        goal_embedding = self.model.encode(' '.join(goal_keywords), convert_to_tensor=True)
        post_embeddings = self.model.encode(titles_and_comments, convert_to_tensor=True)

        goal_embedding = goal_embedding / goal_embedding.norm()
        post_embeddings = post_embeddings / post_embeddings.norm(dim=1, keepdim=True)

        dimension = post_embeddings.size(1)
        index = faiss.IndexFlatIP(dimension)

        if GPU is not None:
            faiss_res = faiss.StandardGpuResources()  
            index = faiss.index_cpu_to_gpu(faiss_res, GPU, index)

        index.add(post_embeddings.cpu().detach().numpy())
        D, top_indices = index.search(goal_embedding.cpu().detach().numpy().reshape(1, -1), len(posts))

        # Filter out posts with a score less than 0.7
        filtered_indices = [i for i, score in zip(top_indices.flatten(), D.flatten()) if score >= 0.7]
        
        sorted_posts = [posts[i] for i in filtered_indices if i < len(posts)]
        print(sorted_posts)
        sorted_post_embeddings = post_embeddings[filtered_indices] if filtered_indices else []

        return sorted_posts, sorted_post_embeddings, index



# Example usage
model_name = 'BAAI/bge-base-en-v1.5'
dataset = PostDataset()
model = SubredditProcessor.load_model(model_name)
scraper = SubredditScraper()
link_sorter = LinkSorter(model, dataset, scraper)

link_sorter.process_popular_subreddits(1000)


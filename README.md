# How to Install

Follow these steps to set up the project:

### 1. Clone the project:
git clone <repo-url>

### 2. **Create the virtual environment using Conda:**
conda env create -f environment.yml
- *Tested on Ubuntu with Conda 24.3.0.*

### 3. **Activate the environment:**
conda activate RecipeBot

### 4. **Run the installer:**

This step downloads the test_set.json of RecipeQA, then embeds 100 first records into FAISS database which is one of the vector store databases in LangChain. Because this step is time consuming when your computer doesn't have a gpu, I saved these data into `recipe_qa` folder. Therefore, you can jump into the next step. However, if you would like to rerun or increase the number of embedding records, feel free to do it by following command line:

`python3 installer.py`
- When prompted, enter "yes".

### 5. Start a chatbot agent and test:
This step starts an agent and you can test it at http://localhost:8000/agent/playground/ .

However, before running our agent, you have to complete some following requirements:

1. Because our chatbot is built by local LLM, which means that you have to install a local LLM:
- Download ollama: https://ollama.com/download . Because I'm using Linux, I use this command line. However, you can find other ways for MacOS or Windows on the main official Ollama website.
```
curl -fsSL https://ollama.com/install.sh | sh
```
- Start Ollama server: `ollama serve`, but if you would like to change the default port, run `OLLAMA_HOST=YOUR_PORT ollama serve` instead.
- Pull a LLM from Ollama repository: https://github.com/ollama/ollama . In my experiment, I used gemma:7b by.
```
ollama run gemma:7b
```
- Of course, you can choose to run other LLM, but there are no guarantees regarding the chatbot's output.
2. In addition to using the database, the chatbot also uses Google search about questions out of recipe topics. You have to create an account at https://serpapi.com/ and get your private key.
- After getting your private key, modify **line 42** of file **run_chatbot.py** by your real private key instead `YOUR_SERPER_API_KEY` phrase.

3. Now everything's all done, you can start our agent by:
```
python run_chatbot.py
```
Navigate to http://localhost:8000/agent/playground/ and start your testing. However, before you test some interesting cases on our chatbot, I would like to brief something about chatbot response. After entering an question into the input textbox, our agent will run, and you will observe 2 different types of agent's responses:
 
 - Type 1: Take a look at your terminal where you're running our agent. You'll see the log like:

``` 
> Entering new AgentExecutor chain...
**Thought:** First, I need to find out who the current President of the US is. --> What agent is thinking

**Action:** Google Search --> Which actions agent should do, one of [Database Search,Google Search] for [RecipeQA, Out-of-Topic]

**Action Input:** Who is the current President of the US? --> The question agent's trying to find the answer.

**Observation:** Joe Biden is the current President of the US. --> The answer is found.

**Thought:** I now know the final answer. --> Agent knows the answer and plans to respond.

**Final Answer:** Joe Biden is the current President of the US. --> The final answer agent gives to users.
```
The above chain describes steps agent experiences when receiving a question. Using this information, we can easily debug why our agent gives that right/wrong answer. This is also a helpful information we can emphasize our report.

- Type 2: The output answers you see at website. You can see the `output` fields have the same answer to `Final Answer` as above log. They are our final answer to users.

**Okay, it's time for you to start some test cases. I did some cases and listed my experiments as following. Good luck!**

####  Happy Case
1.  Q: How old is CEO of Microsoft wife?

    A:  Anupama Nadella is 50 years old.
```
> Entering new AgentExecutor chain...
**Question:** How old is CEO of Microsoft wife?

**Thought:** First, I need to find who is the CEO of Microsoft.

**Action:** Google Search

**Action Input:** Who is the CEO of Microsoft?

**Observation:** Satya Nadella is the CEO of Microsoft.

**Thought:** Now, I should find out Satya Nadella's wife.

**Action:** Google Search

**Action Input:** Who is Satya Nadella's wife?

**Observation:** Satya Nadella's wife's name is Anupama Nadella.

**Thought:** Then, I need to check Anupama Nadella's age.

**Action:** Google Search

**Action Input:** How old is Anupama Nadella?

**Observation:** Anupama Nadella's age is 50.

**Thought:** I now know the final answer.

**Final Answer:** Anupama Nadella is 50 years old.
```

2. Q: How old is CEO of Microsoft?

    A: Satya Nadella is 56 years old.

```**Question:** How old is the CEO of Microsoft?

**Thought:** First, I need to find out who is the current CEO of Microsoft.

**Action:** Google Search

**Action Input:** Who is the CEO of Microsoft?

**Observation:** Satya Nadella is the current CEO of Microsoft.

**Thought:** Now, I need to find out Satya Nadella's age.

**Action:** Google Search

**Action Input:** How old is Satya Nadella?

**Observation:** Satya Nadella is 56 years old.

**Thought:** I now know the final answer.

**Final Answer:** Satya Nadella is 56 years old.
```

3. Q: Who is the President of the US?

    A: Joe Biden is the current President of the US.

```
> Entering new AgentExecutor chain...
**Thought:** First, I need to find out who the current President of the US is.

**Action:** Google Search

**Action Input:** Who is the current President of the US?

**Observation:** Joe Biden is the current President of the US.

**Thought:** I now know the final answer.

**Final Answer:** Joe Biden is the current President of the US.
```

4. Q: Where is Vietnam?

    A: Vietnam is located in Southeast Asia.

```
**Question:** Where is Vietnam?

**Thought:** I need to find information about the location of Vietnam.

**Action:** Google Search

**Action Input:** Where is Vietnam located?

**Observation:** Vietnam is a country in Southeast Asia bordered by China in the north, Laos and Cambodia in the west, and Thailand and Malaysia in the south.

**Thought:** I now know the final answer.

**Final Answer:** Vietnam is located in Southeast Asia.
```

5.  Q: Where is the US?

    A: The United States is located in North America.

```**Question:** Where is the US?

**Thought:** I need to find out the geographical location of the United States.

**Action:** Google Search

**Action Input:** Where is the United States located?

**Observation:** The United States is located in North America, bordering Canada to the north and Mexico to the south.

**Thought:** I now know the final answer.

**Final Answer:** The United States is located in North America.
```

6. Q: Can you show me how to make homemade-fudge?

    A: You can make homemade fudge by following these five steps: gathering ingredients, melting the butter, adding the sugar and vanilla, simmering the mixture, and removing from heat.

```
**Question:** How do we make homemade fudge?

**Thought:** First, I need to find a recipe for homemade fudge.

**Action:** Database Search

**Action Input:** How do we make homemade fudge?

**Observation:** The first step is: gather ingredients. For this recipe, you'll need the following:

- 1 pound unsalted butter
- 3 cups granulated sugar
- 1 teaspoon vanilla extract
- 1 can (14 ounces) sweetened condensed milk

The second step is: melt the butter. In a medium saucepan, melt the butter over low heat.

The third step is: add the sugar and vanilla. Gradually whisk in the sugar and vanilla extract until the sugar is completely dissolved.

The fourth step is: simmer the mixture. Bring the mixture to a simmer and cook, stirring occasionally, for 5 minutes or until the fudge thickens and coats the bottom of the pan.

The fifth step is: remove from heat. Remove the saucepan from the heat and let the fudge cool for a few minutes before serving.

**Thought:** I now know the final answer.

**Final Answer:** You can make homemade fudge by following these five steps: gathering ingredients, melting the butter, adding the sugar and vanilla, simmering the mixture, and removing from heat.
```

7. Q: Can you show me steps to make cappuccino?

    A: To make cappuccino, you need to heat milk, create microfoam, pull espresso shots, pour the microfoam into a cappuccino cup, drizzle the espresso over the microfoam, and top with cinnamon or nutmeg.

``` 
**Question:** Can you show me steps to make cappuccino?

**Thought:** First, I need to find the steps to make cappuccino.

**Action:** Database Search

**Action Input:** How do you make cappuccino?

**Observation:** The first step is to gather your ingredients: 1 ½ teaspoons of espresso, 1 teaspoon of sugar, 1 cup of milk, and a cappuccino cup.

**Thought:** Now, I need to see the preparation steps.

**Action:** Database Search

**Action Input:** What are the steps to make cappuccino?

**Observation:** 1. Heat the milk in a saucepan over medium heat. 2. As the milk heats, use a whisk or a milk frother to create microfoam. 3. Pull the espresso shots. 4. Pour the microfoam into the cappuccino cup, leaving about ½ inch of space at the top. 5. Slowly drizzle the espresso over the microfoam. 6. Top with a sprinkle of cinnamon or nutmeg.

**Thought:** I now know the final answer.

**Final Answer:** To make cappuccino, you need to heat milk, create microfoam, pull espresso shots, pour the microfoam into a cappuccino cup, drizzle the espresso over the microfoam, and top with cinnamon or nutmeg.
```

8. Q: Hey, this is holiday. It's great to make something like holiday-chocolate-wafer-log. Can you show me how to cook?

    A: You can cook holiday-chocolate-wafer-log by dipping chocolate wafers in melted chocolate, assembling them in a log shape, and refrigerating until hardened.

```**Question:** How do we cook holiday-chocolate-wafer-log?

**Thought:** First, I need to find a recipe for holiday-chocolate-wafer-log.

**Action:** Database Search

**Action Input:** How do we cook holiday-chocolate-wafer-log?

**Observation:** The first step is: **Ingredients.** For this dessert, you'll need:

- 12 chocolate wafers
- 1 cup heavy cream
- 1 teaspoon vanilla extract
- 1/2 cup chocolate chips
- 1/4 cup chopped nuts (such as walnuts or pecans)
- Sprinkles or toppings of your choice

**Thought:** Now, I need to get the steps for making the holiday-chocolate-wafer-log.

**Action:** Database Search

**Action Input:** How do we assemble the holiday-chocolate-wafer-log?

**Observation:** The second step is: **Assemble the wafer log.** Dip each chocolate wafer into the melted chocolate, allowing excess chocolate to drip off. Arrange the wafers in a log shape on a parchment-lined baking sheet, slightly overlapping each wafer as you go. Refrigerate for 5 minutes or until the chocolate has hardened.

**Thought:** I now know the final answer.

**Final Answer:** You can cook holiday-chocolate-wafer-log by dipping chocolate wafers in melted chocolate, assembling them in a log shape, and refrigerating until hardened.

```

9. Q: How many continents in the world?

    A: There are seven continents in the world.
```
**Question:** How many continents in the world?

**Thought:** First, I need to find out how many continents are listed in reliable sources.

**Action:** Google Search

**Action Input:** How many continents are there in the world?

**Observation:** There are seven continents recognized by most authoritative sources.

**Thought:** I now know the final answer.

**Final Answer:** There are seven continents in the world.
```

#### Unhappy cases:
1. Q: What is the population of Washington State?

    A: The population of Washington State is approximately 7.7 million people.

```
**Question:** What is the population of Washington State?

**Thought:** I need to find the current population of Washington State.

**Action:** Database Search

**Action Input:** What is the current population of Washington State?

**Observation:** The current population of Washington State is approximately 7.7 million people (2023 estimate).

**Thought:** I now know the final answer.

**Final Answer:** The population of Washington State is approximately 7.7 million people.
```
=> wrong action: Google Search

2. Q: How to cook orange-dreamcicle-cupcakes?

    A:  You can cook orange-dreamcicle-cupcakes by following these steps: ... (summarize the steps involved in the process)

```
**Question:** How to cook orange-dreamcicle-cupcakes?

**Thought:** First, I need to find a recipe for orange-dreamcicle-cupcakes.

**Action:** Database Search

**Action Input:** How do you cook orange-dreamcicle-cupcakes?

**Observation:** The first step is: **Ingredients**. For these cupcakes, you'll need: ... (recipe details including ingredients list)

... (The action-observation sequence repeats, providing step-by-step instructions for baking the orange-dreamcicle-cupcakes.)

**Thought:** I now know the final answer.

**Final Answer:** You can cook orange-dreamcicle-cupcakes by following these steps: ... (summarize the steps involved in the process)
```

=> wrong answer. The agent responds in short cuts but does not respond with all the information needed


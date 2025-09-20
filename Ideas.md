# Smart Bee Colony Monitor

Bees are essential to the environmental structure of our world, as well as sustaining humanity.  
Not only are they part of the food chain, they also pollinate as both specialists and generalists.  
**1/3 of our food source depends on bee pollination.**

---

## Dataset  
[Kaggle - Beehive Sounds](https://www.kaggle.com/datasets/annajyang/beehive-sounds/data)

---

## Model Design

**Input:**  
- Audio file (file name → 60s clip)  
- Numeric data  

**Features:**  
- device  
- hive number  
- date  
- hive temperature  
- hive humidity  
- hive pressure  
- weather temperature  
- weather humidity  
- weather pressure  
- wind speed  
- gust speed  
- weather ID  
- cloud coverage  
- rain  
- latitude  
- longitude  
- time  
- frames  

**Output:**  
- queen presence  
- queen acceptance  
- queen status  
- target (or beehive health as per metric)  

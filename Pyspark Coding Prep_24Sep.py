# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
#Given a DataFrame of transactions (user_id, ts, amount), compute total amount per user for transactions in the last 7 days.

transactions_data = [
    # user_id, ts, amount

    (101, "2026-09-22 09:30:00", 500.00),
    (101, "2026-09-20 14:15:00", 750.00),
    (101, "2026-09-16 10:00:00", 300.00),
    (101, "2026-09-14 18:30:00", 900.00),   # Outside 7 days

    (102, "2026-09-21 11:45:00", 1000.00),
    (102, "2026-09-18 16:20:00", 250.00),
    (102, "2026-09-15 08:30:00", 450.00),
    (102, "2026-09-10 12:00:00", 700.00),   # Outside 7 days

    (103, "2026-09-22 07:10:00", 125.50),
    (103, "2026-09-19 19:40:00", 375.50),
    (103, "2026-09-17 13:25:00", 600.00),

    (104, "2026-09-13 09:00:00", 800.00),   # Outside 7 days
    (104, "2026-09-12 17:30:00", 200.00),   # Outside 7 days

    (105, "2026-09-22 10:15:00", None),     # Null amount
    (105, "2026-09-18 15:00:00", 950.00),

    (106, None,                  400.00),    # Null timestamp
]

from pyspark.sql  import SparkSession
from pyspark.sql.functions import col,current_date,date_sub,sum

spark=SparkSession.builder.appName("Spark DataFrames").getOrCreate() 

trans_df=spark.createDataFrame(transactions_data,["user_id", "ts", "amount"])
filter_df=trans_df.filter(col("ts") >= date_sub(current_date(),7))
filter_df.groupBy("user_id").agg(sum("amount").alias("total_amount")).show()


# COMMAND ----------

#Write a DataFrame to Parquet partitioned by date, targeting file sizes ~256 MB, then compact small files post-write.

filter_df.write.partitionBy("ts").mode("append").parquet("<path>")

# COMMAND ----------

orders_data = [
    (1001, 101, "2026-09-20", 500.0),
    (1002, 102, "2026-09-20", 2000.0),
    (1003, 103, "2026-09-21", 1500.0),
    (1004, 101, "2026-09-21", 700.0),
    (1005, 104, "2026-09-21", 600.0),

    (1006, 102, "2026-09-22", 1800.0),
    (1007, 105, "2026-09-22", 300.0),
    (1008, 101, "2026-09-22", 1000.0),

    (1009, 106, "2026-09-22", 2500.0),
    (1010, 103, "2026-09-23", 1000.0),

    (1011, 999, "2026-09-23", 900.0),
    (1012, 102, "2026-09-23", 1200.0)
]

orders_schema = [
    "order_id",
    "customer_id",
    "order_date",
    "amount"
]

customers_data = [
    (101, "John",  "North"),
    (102, "Alice", "North"),
    (103, "David", "North"),

    (104, "Mary",  "South"),
    (105, "James", "South"),
    (106, "Steve", "South"),

    (107, "Chris", "East"),
    (108, "Priya", "West")
]

customers_schema = [
    "customer_id",
    "customer_name",
    "region"
]
from pyspark.sql.functions import dense_rank,count
from pyspark.sql.window import Window

orders_df=spark.createDataFrame(orders_data,orders_schema)
cust_df=spark.createDataFrame(customers_data,customers_schema)

#Fetch data to match below conditions - 
#1. Retrieve the top 3 customers by total order amount for each region.
#2. Include only customers who have placed at least 5 orders. 
#3. Ensure that customers with no matching orders are excluded. ***/

join_df=orders_df.join(cust_df,on="customer_id",how="inner")
total_df=join_df.groupBy("region","customer_id").agg(sum("amount").alias("Total_Amount"))
rank_df = total_df.withColumn("rank",dense_rank().over(Window.partitionBy("region").orderBy(col("Total_Amount").desc())))
#rank_df.show()
rank_df.filter(col("rank") < 3).select("region","customer_id","Total_Amount").show()

# COMMAND ----------

from pyspark.sql.functions import count, col
#2. Include only customers who have placed at least 5 orders.

join_df.groupBy("customer_id").agg(count(col("order_id")).alias("Order_count")).filter(col("Order_count")>=2).show()

# COMMAND ----------

orders_df.createOrReplaceTempView("orders")
cust_df.createOrReplaceTempView("customers") 

spark.sql("with join_table as (select c.region,c.customer_id,sum(o.amount) as Total_amount from orders o inner join customers c on o.customer_id=c.customer_id group by c.region,c.customer_id) \
    ,rank_table as (select *, dense_rank() over(partition by region order by total_amount desc) as rank from join_table) \
          select * from rank_table where rank <3 ").show()

# COMMAND ----------


#2. Include only customers who have placed at least 5 orders.
spark.sql("select o.customer_id,count(o.order_id) as total_order from orders o inner join customers c on o.customer_id=c.customer_id group by o.customer_id having count(o.order_id) >= 2").show()

# COMMAND ----------

# 3.Ensure that customers with no matching orders are excluded.



# COMMAND ----------

# Data Clean:

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import re

spark=SparkSession.builder.appName("Clean_data_Using_regexp").getOrCreate()

data = [
    ("Ani", "32, Smoky Street, Bangalore 560066 {8971212121}"),
    ("Saket", "111, Puri Sea View Road, Buvaneswar {}"),
    ("Gopal", "{321456987} 560016")
]

data_df=spark.createDataFrame(data,["name","address_phone"])
data_df.show(truncate=False)

# COMMAND ----------

import re

x = re.search(r"\{(\d*)\}","32, Smoky Street, Bangalore 560066 {8971212121}")
print(x.group(0),x.group(1))


# COMMAND ----------

def extract_phone_num(text):
    phone_match=re.search(r"\{(\d*)\}",text)
    phone=phone_match.group(1) if phone_match else ""
    
    clean_add=re.sub(r"\{(\d*)\}","",text)
    address=clean_add
    return address,phone

#print(extract_phone_num("32, Smoky Street, Bangalore 560066 {8971212121}"))
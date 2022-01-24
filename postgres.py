import psycopg2
import pandas as pd


def get_group_details(groupID):
    try:

        connection = psycopg2.connect(
            user="dev",
            password="dev1234",
            host="127.0.0.1",
            port="5432",
            database="test3",
        )

        print("Using Python variable in PostgreSQL select Query")

        postgreSQL_select_Query = "select * from " + '"group"' + f" where id ={groupID}"
        df = pd.read_sql(postgreSQL_select_Query, con=connection)
        print(df)
        return df
    except (Exception, psycopg2.Error) as error:
        print("Error fetching data from PostgreSQL table", error)

    finally:
        # closing database connection
        if connection:

            connection.close()
            print("PostgreSQL connection is closed \n")


get_group_details(1)

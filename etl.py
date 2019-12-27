import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql import Window
from pyspark.sql import functions as F


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """
    Description:
        Create a Spark session to process the data.
    
    Arguments:
        None
        
    Returns:
        spark -- The Spark session.
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Description: 
        Load JSON input from song_data through input_data path,
        process the data to extract song_table and artists_table, 
        and subsequently store the queried data to parquet files.
    
    Arguments:
        spark = The Spark session.
        input_data = song_data path to be processed.
        output_data = path to store the output (parquet files).
        
    Returns:
        None
    
    Output:
        songs_table = songs_table parquet file stored in output_data path.
        artists_table = artists_table parquet file stored in output_data path.
    """
    # get filepath to song data file
    song_data = input_data + "song_data"
    
    # read song data file
    print("\n reading song data file \n")
    df = spark.read.json(song_data)
    df.createOrReplaceTempView("songs_data_table")

    # extract columns to create songs table
    print("\n extracting columns to create songs table \n")
    songs_table = spark.sql("""
                                SELECT DISTINCT song_id, title, artist_id, year, duration
                                FROM songs_data_table
                                ORDER BY song_id
                            """).dropDuplicates()
    
    
    # write songs table to parquet files partitioned by year and artist
    print("\n writing songs table to parquet files \n")
    songs_table_parquet_path = output_data + "songs_table_parquet"
    songs_table.write.mode("overwrite").partitionBy("year", "artist_id").parquet(songs_table_parquet_path)
    

    # extract columns to create artists table
    print("\n extracting columns to create artists table \n")
    artists_table = spark.sql("""
                                SELECT  DISTINCT
                                        artist_id        AS artist_id,
                                        artist_name      AS name,
                                        artist_location  AS location,
                                        artist_latitude  AS latitude,
                                        artist_longitude AS longitude
                                FROM songs_data_table
                                ORDER BY artist_id
                            """).dropDuplicates()
    
    # write artists table to parquet files
    print("\n writing artists table to parquet files \n")
    artists_table_parquet_path = output_data + "artists_table_parquet"
    artists_table.write.mode("overwrite").parquet(artists_table_parquet_path)


def process_log_data(spark, input_data, output_data):
    """
    Description:
        Load JSON input from log_data through input_data path,
        process the data to extract users_table, time_table,songplays_table, 
        and store the queried data to parquet files.
        
    Arguments:
        spark = The Spark session.
        input_data = log_data path to be processed.
        output_data = path to store the output (parquet files).
        
    Returns:
        None
    
    Output:
        users_table = users_table parquet files stored in output_data path.
        time_table = time_table parquet files stored in output_data path.
        songplayes_table = songplays_table parquet files stored in output_data path.
    """
    # get filepath to log data file
    log_data = input_data + "log_data"

    # read log data file
    print("\n reading log data file \n")
    df_log = spark.read.json(log_data)
    
    # filter by actions for song plays
    df_filtered = df_log.filter(df_log.page == 'NextSong')

    # extract columns for users table
    print("\n extracting columns to create users table \n")
    df_filtered.createOrReplaceTempView("log_data_table")
    users_table = spark.sql("""
                                SELECT  DISTINCT userId    AS user_id,
                                                 firstName AS first_name,
                                                 lastName  AS last_name,
                                                 gender,
                                                 level
                                FROM log_data_table
                                ORDER BY user_id
                            """).dropDuplicates(['user_id'])
    
    # write users table to parquet files
    print("\n writing users table to parquet files \n")
    users_table_parquet_path = output_data + "users_table_parquet"
    users_table.write.mode("overwrite").parquet(users_table_parquet_path)

    # create timestamp column from original timestamp column
    print("\n creating timestamp column from original timestamp column \n")
    from pyspark.sql import types as T
    get_timestamp = udf(lambda ts: datetime.fromtimestamp(ts / 1000.0), T.TimestampType())
    df_filtered = df_filtered.withColumn("timestamp", get_timestamp("ts"))
    
    # create datetime column from original timestamp column
    print("\n creating datetime column from original timestamp column \n")
    get_datetime = udf(lambda ts: datetime.strftime(ts, '%Y-%m-%d %H:%M:%S'))
    df_filtered = df_filtered.withColumn("datetime", get_datetime("timestamp"))
    
    # extract columns to create time table
    print("\n extracting columns to create time table \n")
    df_filtered.createOrReplaceTempView("log_data_table")
    time_table = spark.sql("""
                            SELECT DISTINCT datetime              AS start_time,
                                            hour(timestamp)       AS hour,
                                            day(timestamp)        AS day,
                                            weekofyear(timestamp) AS week,
                                            month(timestamp)      AS month,
                                            year(timestamp)       AS year,
                                            dayofweek(timestamp)  AS weekday
                            FROM log_data_table
                            ORDER BY start_time
                            """).dropDuplicates()
    
    # write time table to parquet files partitioned by year and month
    print("\n writing time table to parquet files \n")
    time_table_parquet_path = output_data + "time_table_parquet"
    time_table.write.mode("overwrite").partitionBy("year", "month").parquet(time_table_parquet_path)

    # read in song data to use for songplays table
    print("\n read in song data to use for songplays table \n")
    song_data = input_data + "song_data"
    df_sd = spark.read.json(song_data)
    df_log_song_joined = df_filtered.join(df_sd, (df_filtered.artist == df_sd.artist_name) & \
                                              (df_filtered.song == df_sd.title) & \
                                              (df_filtered.length == df_sd.duration))

    # extract columns from joined song and log datasets to create songplays table
    print("\n extracting columns from joined song and log datasets to create songplays table \n")
    df_log_song_joined = df_log_song_joined.withColumn("songplay_id",F.row_number().over(Window.orderBy("artist_id")))
    df_log_song_joined.createOrReplaceTempView("songplays_table_temp")
    songplays_table = spark.sql("""
                                SELECT  DISTINCT
                                        songplay_id,
                                        datetime  AS start_time,
                                        userId    AS user_id,
                                        level,
                                        song_id,
                                        artist_id,
                                        sessionId AS session_id,
                                        location,
                                        userAgent AS user_agent
                                FROM songplays_table_temp
                                """).dropDuplicates()

    # write songplays table to parquet files partitioned by year and month
    print("\n writing songplays table to parquet files partitioned by year and month \n")
    songplays_table_parquet_path = output_data + "songplays_table_parquet"
    songplays_table.write.mode("overwrite").partitionBy("year", "month").parquet(songplays_table_parquet_path)


def main():
    """
    Description:
        This function is used to create the spark session, 
        define variables for inut_data and output_data paths.
        The defined vriables are used to Load JSON input data (song_data and log_data),
        process the data to extract songs_table, artists_table,
        users_table, time_table, songplays_table,
        and store the queried data to parquet files to output_data path.
        
    Arguments:
        None
        
    Output:
        None
    """
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://d-lake-store/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()

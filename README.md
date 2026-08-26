```python
import seaborn as sns
import pandas as pd
import tabula
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import chisquare, monte_carlo_test
from statsmodels.stats.multitest import multipletests
```

# Exploring randomness in lottery drawings

Lottery drawings are designed to be random and impossible to predict. This notebook/project started out of mere curiosity about how trully random are lottery drawings in practice. To explore this question, I focus on Powerball, the largest and best-known lottery games in the United States, and examine how closely its observed number frequencies match the uniform distributions expected from a fair random process.

## Powerball format

Each Powerball drawing produces five distinct white-ball numbers selected without replacement from one pool and one red Powerball number selected from a separate pool. In the current format, the white-ball range is `1–69` and the red-ball range is `1–26`, but the range has varied historcally.

`Power Play` is a separately generated multiplier and does not affect which white or red numbers are selected. It is therefore excluded from the number-frequency analysis. `Double Play` uses the same white- and red-ball ranges but is a separate drawing, so its results are analyzed independently from the main Powerball results. Because the allowable number ranges have changed historically, drawings from different game formats are also analyzed separately.

The historical drawing results used in this analysis are available from the [Florida Lottery Powerball history](https://files.floridalottery.com/exptkt/pb.pdf).

# Data cleaning and sanitation


```python
# fetch the latest updated data
!curl https://files.floridalottery.com/exptkt/pb.pdf -o pb.pdf
```

      % Total    % Received % Xferd  Average Speed  Time    Time    Time   Current
                                     Dload  Upload  Total   Spent   Left   Speed
    100 729.9k 100 729.9k   0      0  1.01M      0                              0



```python
# ignore first page ==> additional content creates a parsing problems
df_lst_raw = tabula.read_pdf("pb.pdf", pages="2-90")
```

    Failed to import jpype dependencies. Fallback to subprocess.
    No module named 'jpype'



```python
l = len(df_lst_raw)
print(l)
```

    178


Why are there 178 elements instad of just 89?


```python
df_lst_raw[0].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Draw Date</th>
      <th>Winning Numbers</th>
      <th>Game</th>
      <th>Unnamed: 0</th>
      <th>Unnamed: 1</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
      <th>Unnamed: 5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7/25/26</td>
      <td>3</td>
      <td>4</td>
      <td>24</td>
      <td>36</td>
      <td>47</td>
      <td>PB 17</td>
      <td>X4</td>
      <td>POWERBALL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7/25/26</td>
      <td>20</td>
      <td>21</td>
      <td>26</td>
      <td>28</td>
      <td>32</td>
      <td>PB 10</td>
      <td>NaN</td>
      <td>POWERBALL DP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7/22/26</td>
      <td>4</td>
      <td>5</td>
      <td>22</td>
      <td>50</td>
      <td>58</td>
      <td>PB 1</td>
      <td>X3</td>
      <td>POWERBALL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7/22/26</td>
      <td>9</td>
      <td>51</td>
      <td>54</td>
      <td>60</td>
      <td>61</td>
      <td>PB 15</td>
      <td>NaN</td>
      <td>POWERBALL DP</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7/20/26</td>
      <td>2</td>
      <td>9</td>
      <td>44</td>
      <td>53</td>
      <td>59</td>
      <td>PB 8</td>
      <td>X2</td>
      <td>POWERBALL</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_lst_raw[1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pages2/91</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



Thats it! `tabula` is reading the page numbers as separate table. Those can be easily excluded by filtering out using the column name


```python
df = df_lst_raw[0]
for _df in  df_lst_raw[2:]:
    if not any("Pages" in c for c in _df.columns):
        df = pd.concat([df, _df], axis=0, ignore_index=True)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Draw Date</th>
      <th>Winning Numbers</th>
      <th>Game</th>
      <th>Unnamed: 0</th>
      <th>Unnamed: 1</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
      <th>Unnamed: 5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7/25/26</td>
      <td>3</td>
      <td>4</td>
      <td>24</td>
      <td>36</td>
      <td>47</td>
      <td>PB 17</td>
      <td>X4</td>
      <td>POWERBALL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7/25/26</td>
      <td>20</td>
      <td>21</td>
      <td>26</td>
      <td>28</td>
      <td>32</td>
      <td>PB 10</td>
      <td>NaN</td>
      <td>POWERBALL DP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7/22/26</td>
      <td>4</td>
      <td>5</td>
      <td>22</td>
      <td>50</td>
      <td>58</td>
      <td>PB 1</td>
      <td>X3</td>
      <td>POWERBALL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7/22/26</td>
      <td>9</td>
      <td>51</td>
      <td>54</td>
      <td>60</td>
      <td>61</td>
      <td>PB 15</td>
      <td>NaN</td>
      <td>POWERBALL DP</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7/20/26</td>
      <td>2</td>
      <td>9</td>
      <td>44</td>
      <td>53</td>
      <td>59</td>
      <td>PB 8</td>
      <td>X2</td>
      <td>POWERBALL</td>
    </tr>
  </tbody>
</table>
</div>



The column names are all messed up, lets fix them.


```python
df = df.rename(columns = {"Draw Date": "date",
                     "Winning Numbers": "white_ball_1",
                     "Game": "white_ball_2",
                     "Unnamed: 0": "white_ball_3",
                     "Unnamed: 1": "white_ball_4",
                     "Unnamed: 2": "white_ball_5",
                     "Unnamed: 3": "red_ball", # Power Ball number
                     "Unnamed: 4": "pp_mult",  # Power Play multiplier
                     "Unnamed: 5": "game"})
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>white_ball_1</th>
      <th>white_ball_2</th>
      <th>white_ball_3</th>
      <th>white_ball_4</th>
      <th>white_ball_5</th>
      <th>red_ball</th>
      <th>pp_mult</th>
      <th>game</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7/25/26</td>
      <td>3</td>
      <td>4</td>
      <td>24</td>
      <td>36</td>
      <td>47</td>
      <td>PB 17</td>
      <td>X4</td>
      <td>POWERBALL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7/25/26</td>
      <td>20</td>
      <td>21</td>
      <td>26</td>
      <td>28</td>
      <td>32</td>
      <td>PB 10</td>
      <td>NaN</td>
      <td>POWERBALL DP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7/22/26</td>
      <td>4</td>
      <td>5</td>
      <td>22</td>
      <td>50</td>
      <td>58</td>
      <td>PB 1</td>
      <td>X3</td>
      <td>POWERBALL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7/22/26</td>
      <td>9</td>
      <td>51</td>
      <td>54</td>
      <td>60</td>
      <td>61</td>
      <td>PB 15</td>
      <td>NaN</td>
      <td>POWERBALL DP</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7/20/26</td>
      <td>2</td>
      <td>9</td>
      <td>44</td>
      <td>53</td>
      <td>59</td>
      <td>PB 8</td>
      <td>X2</td>
      <td>POWERBALL</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.DataFrame'>
    RangeIndex: 2848 entries, 0 to 2847
    Data columns (total 9 columns):
     #   Column        Non-Null Count  Dtype
    ---  ------        --------------  -----
     0   date          2848 non-null   str  
     1   white_ball_1  2848 non-null   int64
     2   white_ball_2  2848 non-null   int64
     3   white_ball_3  2848 non-null   int64
     4   white_ball_4  2848 non-null   int64
     5   white_ball_5  2848 non-null   int64
     6   red_ball      2848 non-null   str  
     7   pp_mult       2077 non-null   str  
     8   game          2848 non-null   str  
    dtypes: int64(5), str(4)
    memory usage: 200.4 KB


Much better, but there's still some problemns. The `pb` and `pp` columns are `str` type and need to be `int`. Before they can be casted, the additinal characters (PB, X) must be removed from each column. The date is also a `str`, so needs to be converted to a date type if date operation are needed.


```python
df['red_ball'] = df['red_ball'].str.replace('PB ', '', regex=False)
df['pp_mult'] = df['pp_mult'].str.replace('X', '', regex=False)
```


```python
df['red_ball'] = df['red_ball'].astype(int)
# the astype method cannot be used directly with the pp column since it contain NaN strings which cannot be
# casted to int directly. Instead we use the to_numeric method with the "coerce" option which forces the casting
# on the valid elements but asigns NaN to the unvalid ones.
df['pp_mult'] = pd.to_numeric(df['pp_mult'], errors="coerce").astype("Int64")
df["date"] = pd.to_datetime(df["date"])
```

    /tmp/ipykernel_231025/528970629.py:6: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
      df["date"] = pd.to_datetime(df["date"])



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>white_ball_1</th>
      <th>white_ball_2</th>
      <th>white_ball_3</th>
      <th>white_ball_4</th>
      <th>white_ball_5</th>
      <th>red_ball</th>
      <th>pp_mult</th>
      <th>game</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2026-07-25</td>
      <td>3</td>
      <td>4</td>
      <td>24</td>
      <td>36</td>
      <td>47</td>
      <td>17</td>
      <td>4</td>
      <td>POWERBALL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2026-07-25</td>
      <td>20</td>
      <td>21</td>
      <td>26</td>
      <td>28</td>
      <td>32</td>
      <td>10</td>
      <td>&lt;NA&gt;</td>
      <td>POWERBALL DP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2026-07-22</td>
      <td>4</td>
      <td>5</td>
      <td>22</td>
      <td>50</td>
      <td>58</td>
      <td>1</td>
      <td>3</td>
      <td>POWERBALL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2026-07-22</td>
      <td>9</td>
      <td>51</td>
      <td>54</td>
      <td>60</td>
      <td>61</td>
      <td>15</td>
      <td>&lt;NA&gt;</td>
      <td>POWERBALL DP</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2026-07-20</td>
      <td>2</td>
      <td>9</td>
      <td>44</td>
      <td>53</td>
      <td>59</td>
      <td>8</td>
      <td>2</td>
      <td>POWERBALL</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.DataFrame'>
    RangeIndex: 2848 entries, 0 to 2847
    Data columns (total 9 columns):
     #   Column        Non-Null Count  Dtype         
    ---  ------        --------------  -----         
     0   date          2848 non-null   datetime64[us]
     1   white_ball_1  2848 non-null   int64         
     2   white_ball_2  2848 non-null   int64         
     3   white_ball_3  2848 non-null   int64         
     4   white_ball_4  2848 non-null   int64         
     5   white_ball_5  2848 non-null   int64         
     6   red_ball      2848 non-null   int64         
     7   pp_mult       2077 non-null   Int64         
     8   game          2848 non-null   str           
    dtypes: Int64(1), datetime64[us](1), int64(6), str(1)
    memory usage: 203.2 KB


Data looks good now, so we can go ahead and save the cleaned version to disk.


```python
df.to_csv("powerball_numbers_cleaned.csv", index=False)
```

## Consistency Checks

In the powerball drawing, the number of the 5 white balls should be in the range `[1-69]`. The red ball (power ball) numbers should be in the range `[1-26]`


```python
assert df[[f"white_ball_{x}" for x in range(1,6)]].min(axis=None) >= 1,  "White ball numbers out of range!"
assert df[[f"white_ball_{x}" for x in range(1,6)]].max(axis=None) <= 69, "White ball numbers out of range!"
```


```python
assert df["red_ball"].min(axis=None) >= 1,  "Red (Power) ball numbers out of range!"
assert df["red_ball"].max(axis=None) <= 26, "Red (Power) ball numbers out of range!"
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    Cell In[18], line 2
          1 assert df["red_ball"].min(axis=None) >= 1,  "Red (Power) ball numbers out of range!"
    ----> 2 assert df["red_ball"].max(axis=None) <= 26, "Red (Power) ball numbers out of range!"


    AssertionError: Red (Power) ball numbers out of range!



```python
df["red_ball"].max(axis=None)
```




    np.int64(39)



But why? Turns out there have been several versions of the powerball with different ball numbering ranges. [Seven](https://www.usamega.com/powerball/statistics) versions in total to be precise. The availbale data starts on January 7, 2009 and cover the last three versions: versions 5, 6 and 7 (the current version).
|version  |  start date  | end date   | white ball range  | red ball range |
|---------|------------- |------------|-------------------|-----------------
|5        | 2009-01-07   | 2012-01-14 |  1-59             | 1-39           |
|6        | 2012-01-18   | 2015-10-03 |  1-59             | 1-35           |
|7        | 2015-10-07   | present    |  1-69             | 1-26           |

The dataset must be split into these cases to avoid skewing the statistical results.


```python
df_v5 = df[ (df["date"] >= "2009-01-07") & (df["date"] <= "2012-01-14")]
assert df_v5["red_ball"].max(axis=None) == 39, "Power ball numbers out of range!"
df_v5.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>white_ball_1</th>
      <th>white_ball_2</th>
      <th>white_ball_3</th>
      <th>white_ball_4</th>
      <th>white_ball_5</th>
      <th>red_ball</th>
      <th>pp_mult</th>
      <th>game</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2544</th>
      <td>2012-01-14</td>
      <td>10</td>
      <td>30</td>
      <td>36</td>
      <td>38</td>
      <td>41</td>
      <td>1</td>
      <td>5</td>
      <td>POWERBALL</td>
    </tr>
    <tr>
      <th>2545</th>
      <td>2012-01-11</td>
      <td>5</td>
      <td>19</td>
      <td>29</td>
      <td>45</td>
      <td>47</td>
      <td>25</td>
      <td>2</td>
      <td>POWERBALL</td>
    </tr>
    <tr>
      <th>2546</th>
      <td>2012-01-07</td>
      <td>3</td>
      <td>21</td>
      <td>24</td>
      <td>38</td>
      <td>39</td>
      <td>24</td>
      <td>5</td>
      <td>POWERBALL</td>
    </tr>
    <tr>
      <th>2547</th>
      <td>2012-01-04</td>
      <td>21</td>
      <td>35</td>
      <td>46</td>
      <td>47</td>
      <td>50</td>
      <td>2</td>
      <td>4</td>
      <td>POWERBALL</td>
    </tr>
    <tr>
      <th>2548</th>
      <td>2011-12-31</td>
      <td>5</td>
      <td>23</td>
      <td>25</td>
      <td>28</td>
      <td>40</td>
      <td>34</td>
      <td>4</td>
      <td>POWERBALL</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_v6 = df[ (df["date"] >= "2012-01-18") & (df["date"] <= "2015-10-03")]
assert df_v6["red_ball"].max(axis=None) == 35, "Power ball numbers out of range!"
df_v6.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>white_ball_1</th>
      <th>white_ball_2</th>
      <th>white_ball_3</th>
      <th>white_ball_4</th>
      <th>white_ball_5</th>
      <th>red_ball</th>
      <th>pp_mult</th>
      <th>game</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2156</th>
      <td>2015-10-03</td>
      <td>6</td>
      <td>26</td>
      <td>33</td>
      <td>44</td>
      <td>46</td>
      <td>4</td>
      <td>2</td>
      <td>POWERBALL</td>
    </tr>
    <tr>
      <th>2157</th>
      <td>2015-09-30</td>
      <td>21</td>
      <td>39</td>
      <td>40</td>
      <td>55</td>
      <td>59</td>
      <td>17</td>
      <td>3</td>
      <td>POWERBALL</td>
    </tr>
    <tr>
      <th>2158</th>
      <td>2015-09-26</td>
      <td>23</td>
      <td>31</td>
      <td>42</td>
      <td>50</td>
      <td>57</td>
      <td>5</td>
      <td>3</td>
      <td>POWERBALL</td>
    </tr>
    <tr>
      <th>2159</th>
      <td>2015-09-23</td>
      <td>8</td>
      <td>29</td>
      <td>41</td>
      <td>51</td>
      <td>58</td>
      <td>5</td>
      <td>2</td>
      <td>POWERBALL</td>
    </tr>
    <tr>
      <th>2160</th>
      <td>2015-09-19</td>
      <td>12</td>
      <td>17</td>
      <td>26</td>
      <td>43</td>
      <td>48</td>
      <td>24</td>
      <td>2</td>
      <td>POWERBALL</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_v7 = df[df["date"] >= "2015-10-07"]
assert df_v7["red_ball"].max(axis=None) == 26, "Power ball numbers out of range!"
df_v7.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>white_ball_1</th>
      <th>white_ball_2</th>
      <th>white_ball_3</th>
      <th>white_ball_4</th>
      <th>white_ball_5</th>
      <th>red_ball</th>
      <th>pp_mult</th>
      <th>game</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2026-07-25</td>
      <td>3</td>
      <td>4</td>
      <td>24</td>
      <td>36</td>
      <td>47</td>
      <td>17</td>
      <td>4</td>
      <td>POWERBALL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2026-07-25</td>
      <td>20</td>
      <td>21</td>
      <td>26</td>
      <td>28</td>
      <td>32</td>
      <td>10</td>
      <td>&lt;NA&gt;</td>
      <td>POWERBALL DP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2026-07-22</td>
      <td>4</td>
      <td>5</td>
      <td>22</td>
      <td>50</td>
      <td>58</td>
      <td>1</td>
      <td>3</td>
      <td>POWERBALL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2026-07-22</td>
      <td>9</td>
      <td>51</td>
      <td>54</td>
      <td>60</td>
      <td>61</td>
      <td>15</td>
      <td>&lt;NA&gt;</td>
      <td>POWERBALL DP</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2026-07-20</td>
      <td>2</td>
      <td>9</td>
      <td>44</td>
      <td>53</td>
      <td>59</td>
      <td>8</td>
      <td>2</td>
      <td>POWERBALL</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_v7["game"].unique().tolist()
```




    ['POWERBALL', 'POWERBALL DP']



Note that `v7` has two different values in the `game` column: `POWERBALL` and `POWERBALL DP`. The former is the plain old powerball draw while the latter refers to the douple play (hence DP) draw. The double play draw was introduced in August 23, 2021. Note that this falls within version 7 so thats why `POWERBALL DP` is missing from the other dataframes. Lets confirm this to make sure we split the data correctly.


```python
print("Unique games in v5:", df_v5["game"].unique().tolist())
print("Unique games in v6:",df_v6["game"].unique().tolist())
print("Earliest date of POWERBALL DP in v7:", df_v7[df_v7["game"] == "POWERBALL DP"]["date"].min())
```

    Unique games in v5: ['POWERBALL']
    Unique games in v6: ['POWERBALL']
    Earliest date of POWERBALL DP in v7: 2021-08-23 00:00:00


The earliest occurence of a `POWERBALL DP` value in the `game` column of `v7` matches our expectation of August 23, 2021. To avoid mixing things up, lets further split the `v7` data based on the game column.


```python
df_v7_pb = df_v7[ df_v7["game"] == "POWERBALL" ]
df_v7_dp = df_v7[ df_v7["game"] == "POWERBALL DP" ]
```


```python
df_v7_pb.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>white_ball_1</th>
      <th>white_ball_2</th>
      <th>white_ball_3</th>
      <th>white_ball_4</th>
      <th>white_ball_5</th>
      <th>red_ball</th>
      <th>pp_mult</th>
      <th>game</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2026-07-25</td>
      <td>3</td>
      <td>4</td>
      <td>24</td>
      <td>36</td>
      <td>47</td>
      <td>17</td>
      <td>4</td>
      <td>POWERBALL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2026-07-22</td>
      <td>4</td>
      <td>5</td>
      <td>22</td>
      <td>50</td>
      <td>58</td>
      <td>1</td>
      <td>3</td>
      <td>POWERBALL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2026-07-20</td>
      <td>2</td>
      <td>9</td>
      <td>44</td>
      <td>53</td>
      <td>59</td>
      <td>8</td>
      <td>2</td>
      <td>POWERBALL</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2026-07-18</td>
      <td>9</td>
      <td>14</td>
      <td>44</td>
      <td>50</td>
      <td>56</td>
      <td>3</td>
      <td>4</td>
      <td>POWERBALL</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2026-07-15</td>
      <td>2</td>
      <td>7</td>
      <td>18</td>
      <td>29</td>
      <td>38</td>
      <td>16</td>
      <td>2</td>
      <td>POWERBALL</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_v7_dp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>white_ball_1</th>
      <th>white_ball_2</th>
      <th>white_ball_3</th>
      <th>white_ball_4</th>
      <th>white_ball_5</th>
      <th>red_ball</th>
      <th>pp_mult</th>
      <th>game</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2026-07-25</td>
      <td>20</td>
      <td>21</td>
      <td>26</td>
      <td>28</td>
      <td>32</td>
      <td>10</td>
      <td>&lt;NA&gt;</td>
      <td>POWERBALL DP</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2026-07-22</td>
      <td>9</td>
      <td>51</td>
      <td>54</td>
      <td>60</td>
      <td>61</td>
      <td>15</td>
      <td>&lt;NA&gt;</td>
      <td>POWERBALL DP</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2026-07-20</td>
      <td>15</td>
      <td>21</td>
      <td>39</td>
      <td>54</td>
      <td>67</td>
      <td>7</td>
      <td>&lt;NA&gt;</td>
      <td>POWERBALL DP</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2026-07-18</td>
      <td>5</td>
      <td>11</td>
      <td>25</td>
      <td>26</td>
      <td>64</td>
      <td>11</td>
      <td>&lt;NA&gt;</td>
      <td>POWERBALL DP</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2026-07-15</td>
      <td>14</td>
      <td>15</td>
      <td>23</td>
      <td>33</td>
      <td>42</td>
      <td>16</td>
      <td>&lt;NA&gt;</td>
      <td>POWERBALL DP</td>
    </tr>
  </tbody>
</table>
</div>



This concludes the data preparation step.

# Statistical Analysis

To begin, lets take a look at the frequency distribution of the white and red balls for each version of the drawing separately.



```python
white_ball_cols = ["white_ball_1","white_ball_2", "white_ball_3", "white_ball_4", "white_ball_5"]
```


```python
fig, axes = plt.subplots(2,2,figsize=(10, 7),layout="constrained")

white_color = "steelblue"
red_color = "crimson"

groups = [
            (axes[0, 0], df_v5,    "Version 5"),
            (axes[0, 1], df_v6,    "Version 6"),
            (axes[1, 0], df_v7_pb, "Version 7 - Powerball"),
            (axes[1, 1], df_v7_dp, "Version 7 - Double Play"),
]

for ax, data, name in groups:
    # White-ball frequency distribution
    sns.histplot(
                data[white_ball_cols].to_numpy().flatten(),
                stat="count",
                discrete=True,
                color=white_color,
                alpha=0.65,
                ax=ax
                )

    # Red-ball frequency distribution
    sns.histplot(
                data["red_ball"],
                stat="count",
                discrete=True,
                color=red_color,
                alpha=0.65,
                ax=ax
                )

    # Format the minimum and maximum dates for the subplot title.
    start_date = data["date"].min().strftime("%Y-%m-%d")
    end_date = data["date"].max().strftime("%Y-%m-%d")

    ax.set_title(f"{name}\n{start_date} to {end_date}")
    ax.set_xlabel("Ball number")
    ax.set_ylabel("Frequency")

# Proxy patches create one shared legend for the entire figure.
legend_handles = [ Patch(facecolor=white_color, alpha=0.65, label="White balls"),
                   Patch(facecolor=red_color, alpha=0.65, label="Red balls")
                 ]

fig.legend(
            handles=legend_handles,
            loc="outside upper center",
            ncols=2,
            title="Ball type"
            )

plt.show()
```


    
![png](README_files/README_42_0.png)
    


The Version 7 histograms appear reasonably uniform by visual inspection, whereas the Version 5 and Version 6 distributions show more noticeable variations. This apparent difference can likely be explained, at least in part, by sample size. Versions 5 and 6 contain considerably fewer drawings than Version 7, so their observed frequencies are expected to fluctuate more strongly.


```python
print(f"Version 5 drawings: {len(df_v5)}")
print(f"Version 6 drawings: {len(df_v6)}")
print(f"Version 7 - Powerball drawings: {len(df_v7_pb)}")
print(f"Version 7 - Powerball drawings: {len(df_v7_dp)}")
```

    Version 5 drawings: 304
    Version 6 drawings: 388
    Version 7 - Powerball drawings: 1385
    Version 7 - Powerball drawings: 771


Visual inspection is a good starting point, but it alone cannot tell us whether the differences in the histograms represent real nonuniformity or ordinary random variation. This is especially important here because the categories contain different numbers of drawings. Categories with fewer drawings will naturally produce noisier-looking histograms.

A standard chi-square test works for the red ball because only one red ball is selected in each drawing. It is not directly applicable to the pooled white-ball counts because the five white balls in each drawing are selected without replacement. Once one white ball has been selected, it cannot be selected again in the same drawing, so the five selections are not independent.

A Monte Carlo goodness-of-fit test handles this by simulating the actual drawing process, including the correct number of drawings, number range, and without-replacement rule. The deviation from uniformity in the observed data is then compared with the deviations found in many simulated fair datasets. Because eight distributions are tested, the Holm correction is also applied to reduce the chance of treating an ordinary random fluctuation as statistically significant.

### Null hypothesis

For each drawing regime and ball type, the null hypothesis is that the balls are drawn fairly and uniformly.

For the red ball, this means that every number in the allowed range has the same probability of being selected. For the white balls, it means that every possible set of five unique numbers is equally likely. As a result, every individual white-ball number has the same probability of appearing in a drawing, even though the five selections within that drawing are not independent. The drawing results are also assumed to be independent from one drawing to the next.

The alternative hypothesis is that at least one ball number has a different probability of being selected than expected under this fair-drawing model.

### Monte Carlo goodness-of-fit procedure

Each drawing regime and ball type is analyzed separately. First, the observed results are converted into a frequency vector,

$$
\mathbf{O} = (O_1, O_2, \ldots, O_N),
$$

where $O_i$ is the number of times ball $i$ appeared and $N$ is the number of possible ball numbers.

For a dataset containing $D$ drawings, the expected count for each red-ball number under a uniform distribution is

$$
E_{\mathrm{red}} = \frac{D}{N}.
$$

Because each drawing contains five white balls, the expected count for each white-ball number is

$$
E_{\mathrm{white}} = \frac{5D}{N}.
$$

The difference between the observed and expected frequencies is summarized using the Pearson discrepancy statistic,

$$
\chi^2_{\mathrm{obs}}=
\sum_{i=1}^{N}
\frac{(O_i-E_i)^2}{E_i},
$$

where $E_i$ is the appropriate expected count for the ball type being analyzed. A larger value of $\chi^2_{\mathrm{obs}}$ means that the observed frequencies are farther from the frequencies expected under uniform drawing.

For each Monte Carlo resample, a complete fair dataset containing the same number of drawings as the observed dataset is generated. For the red ball, one number is selected uniformly from the allowed range in every drawing. For the white balls, five unique numbers are selected without replacement in every drawing. This preserves the dependence among the five white balls.

The simulated drawings are converted into frequency counts, and the same statistic is calculated. Repeating this process $B=10000$ times produces the simulated statistics

$$
\chi^2_1, \chi^2_2, \ldots, \chi^2_B.
$$

Let $b$ be the number of simulations for which

$$
\chi^2_j \geq \chi^2_{\mathrm{obs}}.
$$

The Monte Carlo p-value is then

$$
p_{\mathrm{MC}}=
\frac{b+1}{B+1}.
$$

The p-value tells us how often a fair simulated dataset produces a deviation at least as large as the one found in the observed data. A small p-value means that very few fair simulations were as uneven as the observed data, providing evidence against the null hypothesis. A large p-value means that the observed amount of unevenness is reasonably common under fair random drawing. It does not prove that the distribution is perfectly uniform.

Because every simulated dataset uses the same number of drawings and the same number range as its corresponding observed dataset, this procedure automatically accounts for the differences among drawing regimes. It is particularly useful for the white balls because it preserves the without-replacement rule that is missing from an ordinary Pearson chi-square test based on independent selections.

Eight separate distributions are analyzed, so eight p-values are produced. These p-values are adjusted using the Holm procedure. This correction controls the overall chance of making one or more false-positive conclusions across the eight tests. The Holm correction does not change the simulations or the individual test statistics; it only changes the threshold used to decide whether any of the eight results is statistically significant.

Fortunately, `scipy.stats` provides the `monte_carlo_test` function, which handles the resampling procedure, constructs the null distribution of the statistic, and calculates the Monte Carlo p-value. The observed frequency vector is supplied through the `data` argument, while custom functions specify how fair lottery datasets are generated and how the Pearson discrepancy statistic is calculated.



```python
def compute_chisquare(observed_counts, axis=-1):
    """
    Wrapper function around SciPy's built-in chisquare

    Parameters
    ----------
    observed_counts:
        shape = (n_possible,) OR (batch_size, n_possible)
        
    axis:
        axis of the input along which to compute statistic
    """
    return chisquare(observed_counts, axis=axis).statistic


def make_fair_draws_generator( n_draws,
                                n_possible,
                                balls_per_draw,
                                rng):
    """
    Factory that creates a fair-count generator configured
    for one Powerball distribution.

    Parameters
    ----------
    n_draws:
        Number of drawings in each simulated dataset.

    n_possible:
        Number of possible ball numbers.

    balls_per_draw:
        Number of balls selected in each drawing:
            1 for red balls
            5 for white balls

    rng:
        NumPy random-number generator.

    Returns
    -------
    generate_fair_counts:
        Function with the rvs(size) interface required by
        scipy.stats.monte_carlo_test.
    """

    if n_draws <= 0:
        raise ValueError("n_draws must be positive.")

    if n_possible <= 0:
        raise ValueError("n_possible must be positive.")

    if not 1 <= balls_per_draw <= n_possible:
        raise ValueError("balls_per_draw must be between 1 and n_possible.")
    

    def generate_fair_counts(size):
        """
        Generate simulated count vectors.

        SciPy supplies size in the form:

            (number of simulations, number of categories)
        """

        n_simulations = size[0]

        if balls_per_draw == 1:
            # Generate aggregate counts from n_draws independent red-ball drawings.
            uniform_probabilities = np.full(n_possible, 1/n_possible)
            # Use multinomial because there are n_possible choices, and only 
            # one is selected each draw. An experiment or dataset consists of
            # n_draws and each of these experiments is repeated n_simulations times.
            simulated_counts = rng.multinomial(n=n_draws,
                                               pvals=uniform_probabilities,
                                               size=n_simulations)
        else:
            # Generate n_draws selections without replacement for every simulated dataset.
            simulated_draws = rng.multivariate_hypergeometric(colors = np.ones(n_possible, dtype=int),
                                                              nsample=balls_per_draw,
                                                              size=(n_simulations, n_draws))

            # Input shape:
            # (n_simulations, n_draws, n_possible)
            #
            # Output shape:
            # (n_simulations, n_possible)
            simulated_counts = simulated_draws.sum(axis=1)

        return simulated_counts

    return generate_fair_counts


def run_uniformity_test(data_frame,
                        ball_columns,
                        n_possible,
                        balls_per_draw,
                        rng,
                        n_samples = 10_000,
                        batch_size = 50):
    """
    Test whether one ball distribution is consistent with
    uniform random drawing.
    
    Parameters
    ----------
    data_frame:
        Pandas data frame containing draw data

    ball_columns:
        columns of the data frame containing winning numbers
            - white_ball_{1-5} for white balls draws
            - red_ball for red ball draws

    balls_per_draw:
        Number of balls selected in each drawing:
            - 1 for red balls
            - 5 for white balls
    rng:
        NumPy random-number generator.

    n_samples:
        Number of MC experiments to perform on each dataset

    batch_size:
        The number of Monte Carlo samples to process in each call.
    """
    
    n_draws = len(data_frame)
    
    if n_draws == 0:
        raise ValueError("Empty dataset")

    # raw values
    observed_values = data_frame[ball_columns].to_numpy(dtype=int).flatten()

    observed_counts = np.bincount(observed_values, minlength = n_possible +1)[1:] # exclude zero

    make_fair_draws = make_fair_draws_generator( n_draws,
                                                 n_possible,
                                                 balls_per_draw,
                                                 rng)

    result = monte_carlo_test( data=observed_counts,
                               rvs=make_fair_draws,
                               statistic=compute_chisquare,
                               vectorized=True,
                               n_resamples=n_samples,
                               batch=batch_size,
                               alternative="greater")
    return result
```


```python
def plot_monte_carlo_distribution(result, title, ax=None):
    """
    Plot the Monte Carlo null distribution for one test.

    result:
        Object returned by scipy.stats.monte_carlo_test.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    simulated_chi_sq = np.asarray(result.null_distribution).flatten()
    observed_chi_sq = result.statistic
    pvalue = result.pvalue

    ax.hist(
        simulated_chi_sq,
        bins=40,
        color="steelblue",
        alpha=0.75,
        edgecolor="white"
    )

    ax.axvline(
        observed_chi_sq,
        color="darkred",
        linestyle="--",
        linewidth=2,
        label=fr"Observed $\chi^2={observed_chi_sq:.2f}$"
    )

    # add empty element so that p value appears in legend
    ax.plot(
        [],
        color="darkred",
        linestyle="",
        linewidth=0,
        label=fr"$p={pvalue:.3f}$"
    )

    ax.set_title(title)
    ax.set_xlabel(r"Simulated $\chi^2$")
    ax.set_ylabel("Number of simulations")
    ax.legend()
```

Now we compute the results for each fo the eight distributions and store them in a dataframe and plot the simulated distributions along with their respective $\chi$ and p values


```python
rng = np.random.default_rng(seed = 10)

# Each tuple contains:
# name, dataframe, white-ball maximum, red-ball maximum
datasets = [("v5", df_v5, 59, 39),
            ("v6", df_v6, 59, 35),
            ("v7_pb", df_v7_pb, 69, 26),
            ("v7_dp", df_v7_dp, 69, 26)]

results_dict = { "draw_version":[],
                 "ball_color":[], 
                 "n_draws": [],
                 "chi_sq": [],
                 "p_value": [],}

# collect results from the monte_carlo_test function for plotting later
results_mc_test = []

# perform MC simulations for each dataset
for version, df, n_white, n_red in datasets:

    white_result = run_uniformity_test(df,
                                       white_ball_cols,
                                       n_white,
                                       balls_per_draw=5,
                                       rng=rng)
    results_dict["draw_version"].append(version)
    results_dict["n_draws"].append(len(df))
    results_dict["ball_color"].append("white")
    results_dict["p_value"].append(white_result.pvalue)
    results_dict["chi_sq"].append(white_result.statistic)

    red_result = run_uniformity_test(df,
                                     "red_ball",
                                      n_red,
                                      balls_per_draw=1,
                                      rng=rng)
    results_dict["draw_version"].append(version)
    results_dict["n_draws"].append(len(df))
    results_dict["ball_color"].append("red")
    results_dict["p_value"].append(red_result.pvalue)
    results_dict["chi_sq"].append(red_result.statistic)

    results_mc_test.append((version, "white balls", white_result))
    results_mc_test.append((version, "red balls", red_result))
                  
results_df = pd.DataFrame(results_dict)
```


```python
results_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>draw_version</th>
      <th>ball_color</th>
      <th>n_draws</th>
      <th>chi_sq</th>
      <th>p_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>v5</td>
      <td>white</td>
      <td>304</td>
      <td>79.132895</td>
      <td>0.013399</td>
    </tr>
    <tr>
      <th>1</th>
      <td>v5</td>
      <td>red</td>
      <td>304</td>
      <td>38.276316</td>
      <td>0.457054</td>
    </tr>
    <tr>
      <th>2</th>
      <td>v6</td>
      <td>white</td>
      <td>388</td>
      <td>37.108247</td>
      <td>0.970603</td>
    </tr>
    <tr>
      <th>3</th>
      <td>v6</td>
      <td>red</td>
      <td>388</td>
      <td>32.541237</td>
      <td>0.543246</td>
    </tr>
    <tr>
      <th>4</th>
      <td>v7_pb</td>
      <td>white</td>
      <td>1385</td>
      <td>81.324765</td>
      <td>0.069293</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(4,2,figsize=(10, 10),layout="constrained")
axes = axes.flatten()
for i, (version, color, result) in enumerate(results_mc_test):
    plot_monte_carlo_distribution(result, f"{version} - {color}", axes[i])
```


    
![png](README_files/README_51_0.png)
    


The figure above shows how the observed $\chi^2$ values compare with the distributions produced by the MC simulations. Version 5 white balls stand out the most, with the observed value falling well into the right tail of the simulated distribution and giving an unadjusted p-value of $0.013$. Version 7 Powerball white balls also fall toward the right side, but the result is less unusual, with $p=0.069$. The remaining results are within or below the typical range of the fair simulations, providing no evidence of greater-than-expected unevenness. Next, we apply Holm correction to p-values using SciPy's built-in `multipletests` function.



```python
reject, pvals, _, _ = multipletests(pvals=results_df["p_value"].to_numpy(),
                                    alpha=0.05,
                                    method="holm")

results_df["p_value_holm"] = pvals
results_df["reject_null"] = reject

results_df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>draw_version</th>
      <th>ball_color</th>
      <th>n_draws</th>
      <th>chi_sq</th>
      <th>p_value</th>
      <th>p_value_holm</th>
      <th>reject_null</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>v5</td>
      <td>white</td>
      <td>304</td>
      <td>79.132895</td>
      <td>0.013399</td>
      <td>0.107189</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>v5</td>
      <td>red</td>
      <td>304</td>
      <td>38.276316</td>
      <td>0.457054</td>
      <td>1.000000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>v6</td>
      <td>white</td>
      <td>388</td>
      <td>37.108247</td>
      <td>0.970603</td>
      <td>1.000000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>v6</td>
      <td>red</td>
      <td>388</td>
      <td>32.541237</td>
      <td>0.543246</td>
      <td>1.000000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>v7_pb</td>
      <td>white</td>
      <td>1385</td>
      <td>81.324765</td>
      <td>0.069293</td>
      <td>0.485051</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>v7_pb</td>
      <td>red</td>
      <td>1385</td>
      <td>23.599278</td>
      <td>0.543946</td>
      <td>1.000000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>v7_dp</td>
      <td>white</td>
      <td>771</td>
      <td>70.876265</td>
      <td>0.260774</td>
      <td>1.000000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>v7_dp</td>
      <td>red</td>
      <td>771</td>
      <td>21.511025</td>
      <td>0.665533</td>
      <td>1.000000</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



# Conclusion

Version 5 white balls showed the largest departure from uniformity, with $\chi^2=79.13$ and an unadjusted p-value of $0.013$. However, after accounting for all eight comparisons using the Holm correction, its adjusted p-value increased to $0.107$. None of the eight adjusted p-values was below the significance level of $0.05$, so the null hypothesis was not rejected for any distribution. Overall, the results do not provide statistically significant evidence that any of the ball distributions differs from the fair-drawing model.


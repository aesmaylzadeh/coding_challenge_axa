import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import time
import folium
from folium.plugins import HeatMap
import osmnx as ox  
import networkx as nx 
from math import radians, cos, sin, asin, sqrt
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import zipfile
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors # type: ignore
import sys
from pathlib import Path
from data_utils import load_nypd_csvs, filter_crashes, save_if_missing, merge_monthly_csvs
from shapely.geometry import Point, box

#####################################################################################################################
####################################    LOADING DATA  ###############################################################
#####################################################################################################################
start_code_time = time.time()
path_citybike = Path(r"DATA/citybike")
path_nypd = Path(r"DATA/NYPD_DATA")

df_nypd_all = load_nypd_csvs(path_nypd)
df_bike = filter_crashes(df_nypd_all, "BIKE")
df_ebike = filter_crashes(df_nypd_all, "E-BIKE")

save_if_missing(df_bike, Path("nypd/bike_crashes.csv"))
save_if_missing(df_ebike, Path("nypd/ebike_crashes.csv"))

output_citybike_folder = Path("citybike_merged")
merge_monthly_csvs(path_citybike, output_citybike_folder)


#####################################################################################################################
####################################    BIKE/E-BIKE CRAHES  #########################################################
#####################################################################################################################
# ----------------- BIKE CRASHES -----------------
df_bike = pd.read_csv("nypd/bike_crashes.csv", low_memory=False)
df_bike["CRASH DATE"] = pd.to_datetime(df_bike["CRASH DATE"], errors="coerce")
df_bike_2023 = df_bike[df_bike["CRASH DATE"].dt.year == 2023]

monthly_counts_bikes = (df_bike_2023.groupby(df_bike_2023["CRASH DATE"].dt.month).size().rename("count").reset_index())
monthly_counts_bikes.columns = ["Month", "Crashes"]
avg_crashes_bike = monthly_counts_bikes["Crashes"].mean()
std_crashes_bike = monthly_counts_bikes["Crashes"].std()
monthly_injured_bikes = (df_bike_2023.groupby(df_bike_2023["CRASH DATE"].dt.month)["NUMBER OF CYCLIST INJURED"].sum().reset_index())
monthly_injured_bikes.columns = ["Month", "Injured"]
avg_injured_bike = monthly_injured_bikes["Injured"].mean()
std_injured_bike = monthly_injured_bikes["Injured"].std()
monthly_killed_bikes = (df_bike_2023.groupby(df_bike_2023["CRASH DATE"].dt.month)["NUMBER OF CYCLIST KILLED"].sum().reset_index())
monthly_killed_bikes.columns = ["Month", "Killed"]
avg_killed_bike = monthly_killed_bikes["Killed"].mean()
std_killed_bike = monthly_killed_bikes["Killed"].std()

total_bike_crashes = monthly_counts_bikes["Crashes"].sum()
total_bike_injured = monthly_injured_bikes["Injured"].sum()
total_bike_killed = monthly_killed_bikes["Killed"].sum()

monthly_pct_injured_bike = monthly_injured_bikes["Injured"] / monthly_counts_bikes["Crashes"] * 100
monthly_pct_killed_bike = monthly_killed_bikes["Killed"] / monthly_counts_bikes["Crashes"] * 100
# ----------------- E-BIKE CRASHES -----------------
df_ebike = pd.read_csv("nypd/ebike_crashes.csv", low_memory=False)
df_ebike["CRASH DATE"] = pd.to_datetime(df_ebike["CRASH DATE"], errors="coerce")
df_ebike_2023 = df_ebike[df_ebike["CRASH DATE"].dt.year == 2023]

monthly_counts_ebike = (df_ebike_2023.groupby(df_ebike_2023["CRASH DATE"].dt.month).size().rename("count").reset_index())
monthly_counts_ebike.columns = ["Month", "Crashes"]
avg_crashes_ebike = monthly_counts_ebike["Crashes"].mean()
std_crashes_ebike = monthly_counts_ebike["Crashes"].std()
monthly_injured_ebikes = (df_ebike_2023.groupby(df_ebike_2023["CRASH DATE"].dt.month)["NUMBER OF CYCLIST INJURED"].sum().reset_index())
monthly_injured_ebikes.columns = ["Month", "Injured"]
avg_injured_ebike = monthly_injured_ebikes["Injured"].mean()
std_injured_ebike = monthly_injured_ebikes["Injured"].std()
monthly_killed_ebikes = (df_ebike_2023.groupby(df_ebike_2023["CRASH DATE"].dt.month)["NUMBER OF CYCLIST KILLED"].sum().reset_index())
monthly_killed_ebikes.columns = ["Month", "Killed"]
avg_killed_ebike = monthly_killed_ebikes["Killed"].mean()
std_killed_ebike = monthly_killed_ebikes["Killed"].std()

total_ebike_crashes = monthly_counts_ebike["Crashes"].sum()
total_ebike_injured = monthly_injured_ebikes["Injured"].sum()
total_ebike_killed = monthly_killed_ebikes["Killed"].sum()

monthly_pct_injured_ebike = monthly_injured_ebikes["Injured"] / monthly_counts_ebike["Crashes"] * 100
monthly_pct_killed_ebike = monthly_killed_ebikes["Killed"] / monthly_counts_ebike["Crashes"] * 100
#####################################################################################################################
####################################    BIKE/E-BIKE Fahrten  ########################################################
#####################################################################################################################
citybike_folder = Path("citybike_outputs")
citybike_folder.mkdir(exist_ok=True)
output_csv = citybike_folder / "monthly_rides_classic_vs_electric.csv"

if not output_csv.exists():
    monthly_rides_bike = []
    monthly_rides_ebike = []
    months = []

    for file in sorted(output_citybike_folder.glob("*_merged.csv")):
        month_str = file.stem.split("_")[0]      # e.g., "202301"
        month_num = int(month_str[-2:])          # take last two digits → 01, 02, ... 12
        months.append(month_num)
        
        df_month = pd.read_csv(file, low_memory=False)
        if "rideable_type" in df_month.columns:
            classic_count = (df_month["rideable_type"] == "classic_bike").sum()
            electric_count = (df_month["rideable_type"] == "electric_bike").sum()
        else:
            classic_count = 0
            electric_count = 0

        monthly_rides_bike.append(classic_count)
        monthly_rides_ebike.append(electric_count)

    df_monthly_rides = pd.DataFrame({
        "Month": months,
        "Classic": monthly_rides_bike,
        "Electric": monthly_rides_ebike
    })

    df_monthly_rides.to_csv(output_csv, index=False)
    print(f"✅ Saved monthly rides to {output_csv}")
else:
    print(f"⏩ CSV already exists: {output_csv}")

df_rides = pd.read_csv(output_csv, low_memory=False)
avg_classic = df_rides["Classic"].mean()
avg_electric = df_rides["Electric"].mean()
months = df_rides["Month"]

total_bike_rides = df_rides["Classic"].sum()
total_ebike_rides = df_rides["Electric"].sum()

#####################################################################################################################
####################################    UNFALLQUOTE BERECHNEN  ######################################################
#####################################################################################################################
total_rides_ny = 226000000 # taken from https://www.nyc.gov/html/dot/html/bicyclists/bikestats.shtml
citybike_fraction = (total_bike_rides + total_ebike_rides)/total_rides_ny
unfallquote_bike_ny = (monthly_counts_bikes["Crashes"] / (total_rides_ny/24)) * 1000000 # in ppm
unfallquote_ebike_ny = (monthly_counts_ebike["Crashes"] / (total_rides_ny/24)) * 1000000 # in ppm
avg_unfallquote_bike_ny = unfallquote_bike_ny.mean()
avg_unfallquote_ebike_ny = unfallquote_ebike_ny.mean()
unfallquote_bike_citybike = unfallquote_bike_ny *0.15
unfallquote_ebike_citybike = unfallquote_ebike_ny *0.15
avg_unfallquote_bike_citybike = avg_unfallquote_bike_ny*0.15
avg_unfallquote_ebike_citybike = avg_unfallquote_ebike_ny*0.15

#####################################################################################################################
###########################################    MEMBER/CASUAL  #######################################################
#####################################################################################################################
output_csv_members = citybike_folder / "monthly_rides_member_vs_casual.csv"

if not output_csv_members.exists():
    monthly_rides_member = []
    monthly_rides_casual = []
    months_member = []

    for file in sorted(output_citybike_folder.glob("*_merged.csv")):
        month_str = file.stem.split("_")[0]
        month_num = int(month_str[-2:])
        months_member.append(month_num)
        
        df_month = pd.read_csv(file, low_memory=False)
        if "member_casual" in df_month.columns:
            member_count = (df_month["member_casual"] == "member").sum()
            casual_count = (df_month["member_casual"] == "casual").sum()
        else:
            member_count = 0
            casual_count = 0

        monthly_rides_member.append(member_count)
        monthly_rides_casual.append(casual_count)

    df_monthly_member = pd.DataFrame({
        "Month": months_member,
        "Member": monthly_rides_member,
        "Casual": monthly_rides_casual
    })

    df_monthly_member.to_csv(output_csv_members, index=False)
    print(f"✅ Saved monthly member vs casual rides to {output_csv_members}")
else:
    print(f"⏩ CSV already exists: {output_csv_members}")

df_member = pd.read_csv(output_csv_members)
df_member["Total"] = df_member["Member"] + df_member["Casual"]
df_member["Member_pct"] = df_member["Member"] / df_member["Total"] * 100
df_member["Casual_pct"] = df_member["Casual"] / df_member["Total"] * 100
avg_member = df_member["Member"].mean()
avg_casual = df_member["Casual"].mean()

#####################################################################################################################
###########################################    CRASHES MEMBER/CASUAL  ###############################################
#####################################################################################################################
monthly_citybike_bike_crashes = monthly_counts_bikes["Crashes"] * citybike_fraction
monthly_citybike_ebike_crashes = monthly_counts_ebike["Crashes"] * citybike_fraction
member_fraction = df_member["Member"] / df_member["Total"]
casual_fraction = df_member["Casual"] / df_member["Total"]

monthly_citybike_bike_crashes_member = monthly_citybike_bike_crashes * member_fraction
monthly_citybike_bike_crashes_casual = monthly_citybike_bike_crashes * casual_fraction

monthly_citybike_ebike_crashes_member = monthly_citybike_ebike_crashes * member_fraction
monthly_citybike_ebike_crashes_casual = monthly_citybike_ebike_crashes * casual_fraction
#####################################################################################################################
###########################################    COST OF CRASHES  #####################################################
#####################################################################################################################
cost_of_bike = 1500
cost_of_ebike = 2500
cost_of_injured = 8000

costs_monthly_crash_bike_member =  monthly_citybike_bike_crashes_member * cost_of_bike
costs_monthly_crash_bike_casual =  monthly_citybike_bike_crashes_casual * cost_of_bike

costs_monthly_crash_ebike_member =  monthly_citybike_ebike_crashes_member * cost_of_ebike
costs_monthly_crash_ebike_casual =  monthly_citybike_ebike_crashes_casual * cost_of_ebike

monthly_citybike_bike_injured = monthly_injured_bikes["Injured"] * citybike_fraction
monthly_citybike_ebike_injured = monthly_injured_ebikes["Injured"] * citybike_fraction

monthly_citybike_bike_injured_member = monthly_citybike_bike_injured * member_fraction
monthly_citybike_bike_injured_casual = monthly_citybike_bike_injured * casual_fraction

monthly_citybike_ebike_injured_member = monthly_citybike_ebike_injured * member_fraction
monthly_citybike_ebike_injured_casual = monthly_citybike_ebike_injured * casual_fraction

monthly_citybike_bike_injured_cost_member = monthly_citybike_bike_injured_member * cost_of_injured
monthly_citybike_bike_injured_cost_casual = monthly_citybike_bike_injured_casual * cost_of_injured

monthly_citybike_ebike_injured_cost_member = monthly_citybike_ebike_injured_member * cost_of_injured
monthly_citybike_ebike_injured_cost_casual = monthly_citybike_ebike_injured_casual * cost_of_injured

total_monthly_bike_cost_member = costs_monthly_crash_bike_member + monthly_citybike_bike_injured_cost_member
total_monthly_bike_cost_casual = costs_monthly_crash_bike_casual + monthly_citybike_bike_injured_cost_casual

total_monthly_ebike_cost_member = costs_monthly_crash_ebike_member + monthly_citybike_ebike_injured_cost_member
total_monthly_ebike_cost_casual = costs_monthly_crash_ebike_casual + monthly_citybike_ebike_injured_cost_casual
#####################################################################################################################
#####################################################    PLOTS  #####################################################
#####################################################################################################################
folder_path = Path("figures")
folder_path.mkdir(parents=True, exist_ok=True)
output_folder_figures =  folder_path

########################################    ANZAHL CRASHES BIKE/E-BIKE  #############################################
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
bar_width = 0.25
alpha_sigma = 0.1

# ----------------- Bike Crashes -----------------
axes[0].bar(monthly_counts_bikes["Month"] - bar_width, monthly_counts_bikes["Crashes"], color="lightgreen", edgecolor="lightgreen", width=bar_width, label=f"Bike Crashes (Total = {total_bike_crashes})")
axes[0].bar(monthly_injured_bikes["Month"] , monthly_injured_bikes["Injured"], color="orange", edgecolor="orange", width=bar_width, label=f"Injured Cyclists (Total = {total_bike_injured})")
axes[0].bar(monthly_killed_bikes["Month"] + bar_width, monthly_killed_bikes["Killed"], color="red", edgecolor="red", width=bar_width, label=f"Killed Cyclists (Total = {total_bike_killed})")
axes[0].axhline(avg_crashes_bike, color="green", linestyle="--", linewidth=2)
axes[0].axhline(avg_injured_bike, color="orange", linestyle="--", linewidth=2)
axes[0].axhline(avg_killed_bike, color="red", linestyle="--", linewidth=2)
axes[0].text(12.3, avg_crashes_bike + 1, f"Mean = {avg_crashes_bike:.1f}", color="black", fontsize=12, va="bottom")
axes[0].text(12.3, avg_injured_bike + 1, f"Mean = {avg_injured_bike:.1f}", color="black", fontsize=12, va="bottom")
axes[0].text(12.3, avg_killed_bike + 1, f"Mean = {avg_killed_bike:.1f}", color="black", fontsize=12, va="bottom")
axes[0].set_title("Bike Crashes per Month (2023)", fontsize=16)
axes[0].set_xlabel("Month", fontsize=14)
axes[0].set_ylabel("Number of Crashes", fontsize=14)
axes[0].grid(axis="y", linestyle="--", alpha=0.5)
axes[0].legend(fontsize=12)

# ----------------- E-Bike Crashes -----------------
axes[1].bar(monthly_counts_ebike["Month"] - bar_width/2, monthly_counts_ebike["Crashes"], color="lightgreen", edgecolor="lightgreen", width=bar_width, label=f"E-Bike Crashes (Total = {total_ebike_crashes})")
axes[1].bar(monthly_injured_ebikes["Month"] + bar_width/2, monthly_injured_ebikes["Injured"], color="orange", edgecolor="orange", width=bar_width, label=f"Injured Cyclists (Total = {total_ebike_injured})")
axes[1].bar(monthly_killed_ebikes["Month"] + bar_width, monthly_killed_ebikes["Killed"], color="red", edgecolor="red", width=bar_width, label=f"Killed Cyclists (Total = {total_ebike_killed})")
axes[1].axhline(avg_crashes_ebike, color="green", linestyle="--", linewidth=2)
axes[1].axhline(avg_injured_ebike, color="orange", linestyle="--", linewidth=2)
axes[1].axhline(avg_killed_ebike, color="red", linestyle="--", linewidth=2)
axes[1].text(12.3, avg_crashes_ebike + 1, f"Mean = {avg_crashes_ebike:.1f}", color="black", fontsize=12, va="bottom")
axes[1].text(12.3, avg_injured_ebike + 1, f"Mean = {avg_injured_ebike:.1f}", color="black", fontsize=12, va="bottom")
axes[1].text(12.3, avg_killed_ebike + 1, f"Mean = {avg_killed_ebike:.1f}", color="black", fontsize=12, va="bottom")
axes[1].set_title("E-Bike Crashes per Month (2023)", fontsize=16)
axes[1].set_xlabel("Month", fontsize=14)
axes[1].set_ylabel("Number of Crashes", fontsize=14)
axes[1].grid(axis="y", linestyle="--", alpha=0.5)
axes[1].legend(fontsize=12)

plt.xticks(monthly_counts_bikes["Month"])
for ax in axes:
    ax.set_xlim(-1, 14)
plt.tight_layout()
fig.savefig(output_folder_figures / "bike_ebike_crashes_2023.png", dpi=300, bbox_inches="tight")


########################################    PROZENT CRASHES BIKE/E-BIKE  #############################################

# ----------------- Plot Percentages with Averages -----------------
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
bar_width = 0.25

# ----------------- Bike Percentages -----------------
avg_injured_pct_bike = monthly_pct_injured_bike.mean()
avg_killed_pct_bike = monthly_pct_killed_bike.mean()
axes[0].bar(monthly_counts_bikes["Month"] - bar_width/2, monthly_pct_injured_bike, width=bar_width, color="orange", label="Injured %")
axes[0].bar(monthly_counts_bikes["Month"] + bar_width/2, monthly_pct_killed_bike, width=bar_width, color="red", label="Killed %")
axes[0].axhline(avg_injured_pct_bike, color="orange", linestyle="--", linewidth=2)
axes[0].axhline(avg_killed_pct_bike, color="red", linestyle="--", linewidth=2)
axes[0].text(12.3, avg_injured_pct_bike + 1, f"Mean = {avg_injured_pct_bike:.1f}%", color="black", fontsize=12, va="bottom")
axes[0].text(12.3, avg_killed_pct_bike + 1, f"Mean = {avg_killed_pct_bike:.1f}%", color="black", fontsize=12, va="bottom")
axes[0].set_ylabel("Percentage per Crash [%]", fontsize=14)
axes[0].set_title("Bike Injured/Killed Percentage per Crash (2023)", fontsize=16)
axes[0].grid(axis="y", linestyle="--", alpha=0.5)
axes[0].legend(loc="upper left", fontsize=12)

# ----------------- E-Bike Percentages -----------------
avg_injured_pct_ebike = monthly_pct_injured_ebike.mean()
avg_killed_pct_ebike = monthly_pct_killed_ebike.mean()
axes[1].bar(monthly_counts_ebike["Month"] - bar_width/2, monthly_pct_injured_ebike, width=bar_width, color="orange", label="Injured %")
axes[1].bar(monthly_counts_ebike["Month"] + bar_width/2, monthly_pct_killed_ebike, width=bar_width, color="red", label="Killed %")
axes[1].axhline(avg_injured_pct_ebike, color="orange", linestyle="--", linewidth=2)
axes[1].axhline(avg_killed_pct_ebike, color="red", linestyle="--", linewidth=2)
axes[1].text(12.3, avg_injured_pct_ebike + 1, f"Mean = {avg_injured_pct_ebike:.1f}%", color="black", fontsize=12, va="bottom")
axes[1].text(12.3, avg_killed_pct_ebike + 1, f"Mean = {avg_killed_pct_ebike:.1f}%", color="black", fontsize=12, va="bottom")
axes[1].set_ylabel("Percentage per Crash [%]", fontsize=14)
axes[1].set_title("E-Bike Injured/Killed Percentage per Crash (2023)", fontsize=16)
axes[1].set_xlabel("Month", fontsize=14)
axes[1].grid(axis="y", linestyle="--", alpha=0.5)
axes[1].legend(loc="upper left", fontsize=12)

plt.xticks(monthly_counts_bikes["Month"])
for ax in axes:
    ax.set_xlim(-1, 14)
plt.tight_layout()
fig.savefig(output_folder_figures / "bike_ebike_crashes_pct_2023.png", dpi=300, bbox_inches="tight")


########################################    FAHRTEN/MONAT CITYBIKE #############################################
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# ----------------- Classic Bike -----------------
axes[0].bar(months, df_rides["Classic"], color="skyblue", label=f"Classic Bike Total = ({total_bike_rides:.0f})")
axes[0].axhline(avg_classic, color="blue", linestyle="--", linewidth=2)
axes[0].text(-0.3, avg_classic + 1, f"Mean = {avg_classic:.0f}", color="black", fontsize=12, va="bottom")
axes[0].set_ylabel("Number of Rides", fontsize=14)
axes[0].set_title("Monthly Classic Bike Rides with Average", fontsize=16)
axes[0].grid(axis="y", linestyle="--", alpha=0.5)
axes[0].legend(loc="upper left", fontsize=12)

# ----------------- Electric Bike -----------------
axes[1].bar(months, df_rides["Electric"], color="lightcoral", label=f"Electric Bike Total = ({total_ebike_rides:.0f})")
axes[1].axhline(avg_electric, color="red", linestyle="--", linewidth=2)
axes[1].text(-0.3, avg_electric + 1, f"Mean = {avg_electric:.0f}", color="black", fontsize=12, va="bottom")
axes[1].set_xlabel("Month", fontsize=14)
axes[1].set_ylabel("Number of Rides", fontsize=14)
axes[1].set_title("Monthly Electric Bike Rides with Average", fontsize=16)
axes[1].grid(axis="y", linestyle="--", alpha=0.5)
axes[1].legend(loc="upper left", fontsize=12)

# X-axis ticks
plt.xticks(months)
for ax in axes:
    ax.set_xlim(-1, 14)
plt.tight_layout()
plt.savefig(output_folder_figures / "monthly_rides_classic_vs_electric.png", dpi=300, bbox_inches="tight")

########################################    UNFALLQUOTE [%] #############################################
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# ----------------- Classic Bike -----------------
axes[0].bar(months, unfallquote_bike_ny, color="grey", label="Bike Unfallquote NYC$^1$")
axes[0].bar(months, unfallquote_bike_citybike, color="skyblue", label="Bike Unfallquote (Citybike)")
#axes[0].axhline(avg_unfallquote_bike_ny, color="grey", linestyle="--", linewidth=2)
#axes[0].axhline(avg_unfallquote_bike_citybike, color="skyblue", linestyle="--", linewidth=2)
axes[0].text(-0.9, avg_unfallquote_bike_ny, f"Mean = {avg_unfallquote_bike_ny:.3f}[ppm]", color="black", fontsize=12, va="bottom")
axes[0].text(-0.9, avg_unfallquote_bike_ny -5, f"Fraction Citybike = {citybike_fraction*100:.2f}[%]", color="black", fontsize=12, va="bottom")
#axes[0].text(-0.9, avg_unfallquote_bike_citybike, f"Mean = {avg_unfallquote_bike_citybike:.3f}[ppm]", color="black", fontsize=12, va="bottom")
axes[0].set_ylabel("Unfallquote [ppm]", fontsize=14)
axes[0].set_title("Unfallquote Classic Bike in [ppm](2023)", fontsize=16)
axes[0].grid(axis="y", linestyle="--", alpha=0.5)
axes[0].legend(loc="upper left", fontsize=12)

# ----------------- Electric Bike -----------------
axes[1].bar(months, unfallquote_ebike_ny, color="grey", label="E-Bike Unfallquote NYC$^1$")
axes[1].bar(months, unfallquote_ebike_citybike, color="lightcoral", label="E-Bike Unfallquote (Citybike)")
#axes[1].axhline(avg_unfallquote_ebike_ny, color="grey", linestyle="--", linewidth=2)
#axes[1].axhline(avg_unfallquote_ebike_citybike, color="lightcoral", linestyle="--", linewidth=2)
axes[1].text(-0.9, avg_unfallquote_ebike_ny, f"Mean = {avg_unfallquote_ebike_ny:.3f}[ppm]", color="black", fontsize=12, va="bottom")
axes[1].text(-0.9, avg_unfallquote_ebike_ny -5, f"Fraction Citybike = {citybike_fraction*100:.2f}[%]", color="black", fontsize=12, va="bottom")
#axes[1].text(-0.9, avg_unfallquote_ebike_citybike, f"Mean = {avg_unfallquote_ebike_citybike:.3f}[ppm]", color="black", fontsize=12, va="bottom")
axes[1].set_xlabel("Monat", fontsize=14)
axes[1].set_ylabel("Unfallquote [ppm]", fontsize=14)
axes[1].set_title("Unfallquote E-Bike in [ppm](2023)", fontsize=16)
axes[1].grid(axis="y", linestyle="--", alpha=0.5)
axes[1].legend(loc="upper left", fontsize=12)

plt.xticks(months)
for ax in axes:
    ax.set_xlim(-1, 14)
plt.tight_layout()
plt.savefig(output_folder_figures / "unfallquote_bike_ebike_ny.png", dpi=300, bbox_inches="tight")


########################################    Member/Casual [%] #############################################
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# ---------------- Absolute numbers ----------------
axes[0].bar(df_member["Month"] - 0.15, df_member["Member"], width=0.3, label="Member", color="steelblue")
axes[0].bar(df_member["Month"] + 0.15, df_member["Casual"], width=0.3, label="Casual", color="orange")
axes[0].axhline(avg_member, color="steelblue", linestyle="--", linewidth=2)
axes[0].axhline(avg_casual, color="orange", linestyle="--", linewidth=2)
axes[0].text(-0.9, avg_member, f"Mean = {avg_member:.0f}", color="black", fontsize=12, va="bottom")
axes[0].text(-0.9, avg_casual, f"Mean = {avg_casual:.0f}", color="black", fontsize=12, va="bottom")
axes[0].set_ylabel("Number of Rides")
axes[0].set_title("Monthly Citi Bike Rides: Member vs Casual (2023)")
axes[0].grid(axis="y", linestyle="--", alpha=0.5)
axes[0].legend(loc="upper left", fontsize=12)

# ---------------- Percentages ----------------
axes[1].bar(df_member["Month"], df_member["Member_pct"], label="Member %", color="steelblue")
axes[1].bar(df_member["Month"], df_member["Casual_pct"], bottom=df_member["Member_pct"], label="Casual %", color="orange")

# Add percentage labels inside bars
for i, (m_pct, c_pct) in enumerate(zip(df_member["Member_pct"], df_member["Casual_pct"])):
    axes[1].text(df_member["Month"][i], m_pct / 2, f"{m_pct:.1f}%", ha="center", color="white", fontsize=9)
    axes[1].text(df_member["Month"][i], m_pct + c_pct / 2, f"{c_pct:.1f}%", ha="center", color="white", fontsize=9)

axes[1].set_ylabel("Percentage of Rides [%]")
axes[1].set_title("Monthly Citi Bike Rides: Member vs Casual (Percentage, 2023)")
axes[1].set_ylim(0, 100)
axes[1].legend(loc="upper right", fontsize=12)
axes[1].grid(axis="y", linestyle="--", alpha=0.5)

# Shared X-axis settings
plt.xticks(df_member["Month"])
for ax in axes:
    ax.set_xlim(-1, 14)
axes[1].set_xlabel("Month")

plt.tight_layout()
plt.savefig(output_folder_figures / "monthly_rides_member_vs_casual_combined.png", dpi=300)

########################################    Member/Casual Crashes #############################################
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
bar_width = 0.35

# ----------------- Classic Bike -----------------
axes[0].bar(df_member["Month"] - bar_width/2, monthly_citybike_bike_crashes_member, width=bar_width, label=f"Member Total = {monthly_citybike_bike_crashes_member.sum():.0f}", color="steelblue")
axes[0].bar(df_member["Month"] + bar_width/2, monthly_citybike_bike_crashes_casual, width=bar_width, label=f"Casual Total = {monthly_citybike_bike_crashes_casual.sum():.0f}", color="orange")
axes[0].set_ylabel("Estimated Crashes")
axes[0].set_title("Estimated Citybike Classic Bike Crashes by Member/Casual (2023)")
axes[0].grid(axis="y", linestyle="--", alpha=0.5)
axes[0].legend(loc="upper left", fontsize=12)

for i, (m, c) in enumerate(zip(monthly_citybike_bike_crashes_member, monthly_citybike_bike_crashes_casual)):
    axes[0].text(df_member["Month"][i] - bar_width/2, m + 0.5, f"{m:.0f}", ha="center", color="black", fontsize=9)
    axes[0].text(df_member["Month"][i] + bar_width/2, c + 0.5, f"{c:.0f}", ha="center", color="black", fontsize=9)

# ----------------- E-Bike -----------------
axes[1].bar(df_member["Month"] - bar_width/2, monthly_citybike_ebike_crashes_member, width=bar_width, label=f"Member Total = {monthly_citybike_ebike_crashes_member.sum():.0f}", color="steelblue")
axes[1].bar(df_member["Month"] + bar_width/2, monthly_citybike_ebike_crashes_casual, width=bar_width, label=f"Casual Total = {monthly_citybike_ebike_crashes_casual.sum():.0f}", color="orange")
axes[0].set_ylabel("Estimated Crashes")
axes[1].set_ylabel("Estimated Crashes")
axes[1].set_title("Estimated Citybike E-Bike Crashes by Member/Casual (2023)")
axes[1].set_xlabel("Month")
axes[1].grid(axis="y", linestyle="--", alpha=0.5)
axes[1].legend(loc="upper left", fontsize=12)

for i, (m, c) in enumerate(zip(monthly_citybike_ebike_crashes_member, monthly_citybike_ebike_crashes_casual)):
    axes[1].text(df_member["Month"][i] - bar_width/2, m + 0.5, f"{m:.0f}", ha="center", color="black", fontsize=9)
    axes[1].text(df_member["Month"][i] + bar_width/2, c + 0.5, f"{c:.0f}", ha="center", color="black", fontsize=9)

plt.xticks(df_member["Month"])
for ax in axes:
    ax.set_xlim(-1, 14)

plt.tight_layout()
plt.savefig(output_folder_figures / "citybike_crashes_member_vs_casual.png", dpi=300)

########################################    Costs Crashes #############################################
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

bar_width = 0.35

# ----------------- Classic Bike Costs -----------------
axes[0].bar(df_member["Month"] - bar_width/2, costs_monthly_crash_bike_member, width=bar_width, label=f"Member Total = {costs_monthly_crash_bike_member.sum():,.0f}$", color="steelblue")
axes[0].bar(df_member["Month"] + bar_width/2, costs_monthly_crash_bike_casual, width=bar_width, label=f"Casual Total = {costs_monthly_crash_bike_casual.sum():,.0f}$", color="orange")
axes[0].set_ylabel("Cost of Crashes [USD]")
axes[0].set_title("Monthly Cost of Citybike Classic Bike Crashes (2023)")
axes[0].grid(axis="y", linestyle="--", alpha=0.5)
axes[0].legend(loc="upper left", fontsize=12)

# Add cost labels above bars
for i, (m, c) in enumerate(zip(costs_monthly_crash_bike_member, costs_monthly_crash_bike_casual)):
    axes[0].text(df_member["Month"][i] - bar_width/2, m + 50, f"${m:,.0f}", ha="center", fontsize=9)
    axes[0].text(df_member["Month"][i] + bar_width/2, c + 50, f"${c:,.0f}", ha="center", fontsize=9)

# ----------------- E-Bike Costs -----------------
axes[1].bar(df_member["Month"] - bar_width/2, costs_monthly_crash_ebike_member, width=bar_width, label=f"Member Total = {costs_monthly_crash_ebike_member.sum():,.0f}$", color="steelblue")
axes[1].bar(df_member["Month"] + bar_width/2, costs_monthly_crash_ebike_casual, width=bar_width, label=f"Casual Total = {costs_monthly_crash_ebike_casual.sum():,.0f}$", color="orange")

axes[1].set_ylabel("Cost of Crashes [USD]")
axes[1].set_title("Monthly Cost of Citybike E-Bike Crashes (2023)")
axes[1].set_xlabel("Month")
axes[1].grid(axis="y", linestyle="--", alpha=0.5)
axes[1].legend(loc="upper left", fontsize=12)

# Add cost labels above bars
for i, (m, c) in enumerate(zip(costs_monthly_crash_ebike_member, costs_monthly_crash_ebike_casual)):
    axes[1].text(df_member["Month"][i] - bar_width/2, m + 50, f"${m:,.0f}", ha="center", fontsize=9)
    axes[1].text(df_member["Month"][i] + bar_width/2, c + 50, f"${c:,.0f}", ha="center", fontsize=9)

# X-axis ticks
plt.xticks(df_member["Month"])
plt.tight_layout()
plt.savefig(output_folder_figures / "citybike_crash_costs_member_vs_casual.png", dpi=300)

########################################    Costs Crashes Injured #############################################
total_monthly_bike_cost_member = costs_monthly_crash_bike_member + monthly_citybike_bike_injured_cost_member
total_monthly_bike_cost_casual = costs_monthly_crash_bike_casual + monthly_citybike_bike_injured_cost_casual

total_monthly_ebike_cost_member = costs_monthly_crash_ebike_member + monthly_citybike_ebike_injured_cost_member
total_monthly_ebike_cost_casual = costs_monthly_crash_ebike_casual + monthly_citybike_ebike_injured_cost_casual

# ===================== Plot Monthly Costs =====================
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
bar_width = 0.35

# -------- Bike Costs --------
axes[0].bar(df_member["Month"] - bar_width/2, total_monthly_bike_cost_member, width=bar_width,
            label=f"Member Total = ${total_monthly_bike_cost_member.sum():,.0f}", color="steelblue")
axes[0].bar(df_member["Month"] + bar_width/2, total_monthly_bike_cost_casual, width=bar_width,
            label=f"Casual Total = ${total_monthly_bike_cost_casual.sum():,.0f}", color="orange")
axes[0].set_ylabel("Cost of Crashes + Injured [USD]")
axes[0].set_title("Monthly Total Cost: Classic Bike (2023)")
axes[0].grid(axis="y", linestyle="--", alpha=0.5)
axes[0].legend(loc="upper left", fontsize=12)

# Add cost labels above bars
for i, (m, c) in enumerate(zip(total_monthly_bike_cost_member, total_monthly_bike_cost_casual)):
    axes[0].text(df_member["Month"][i] - bar_width/2, m + 50, f"${m:,.0f}", ha="center", fontsize=9)
    axes[0].text(df_member["Month"][i] + bar_width/2, c + 50, f"${c:,.0f}", ha="center", fontsize=9)

# -------- E-Bike Costs --------
axes[1].bar(df_member["Month"] - bar_width/2, total_monthly_ebike_cost_member, width=bar_width,
            label=f"Member Total = ${total_monthly_ebike_cost_member.sum():,.0f}", color="steelblue")
axes[1].bar(df_member["Month"] + bar_width/2, total_monthly_ebike_cost_casual, width=bar_width,
            label=f"Casual Total = ${total_monthly_ebike_cost_casual.sum():,.0f}", color="orange")
axes[1].set_ylabel("Cost of Crashes + Injured [USD]")
axes[1].set_title("Monthly Total Cost: E-Bike (2023)")
axes[1].set_xlabel("Month")
axes[1].grid(axis="y", linestyle="--", alpha=0.5)
axes[1].legend(loc="upper left", fontsize=12)

# Add cost labels above bars
for i, (m, c) in enumerate(zip(total_monthly_ebike_cost_member, total_monthly_ebike_cost_casual)):
    axes[1].text(df_member["Month"][i] - bar_width/2, m + 50, f"${m:,.0f}", ha="center", fontsize=9)
    axes[1].text(df_member["Month"][i] + bar_width/2, c + 50, f"${c:,.0f}", ha="center", fontsize=9)

# X-axis ticks
plt.xticks(df_member["Month"])
for ax in axes:
    ax.set_xlim(-1, 14)

plt.tight_layout()
plt.savefig(output_folder_figures / "citybike_total_costs_member_vs_casual.png", dpi=300)




end_code_time = time.time()
print(f"⏱ Fertig in {end_code_time - start_code_time:.2f} Sekunden")
plt.show()
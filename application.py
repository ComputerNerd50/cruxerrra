import statistics
from flask import Flask, render_template, request, Response, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import csv
import io
import os
import pandas as pd
from time_1 import *

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecret")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///collections.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Races(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    event = db.Column(db.String(100), nullable=False)
    date = db.Column(db.String(20), nullable=False)
    distance = db.Column(db.Integer, nullable=False)
    time_sec = db.Column(db.Float, nullable=False)
    elevation = db.Column(db.Integer, nullable=False)
    humidity = db.Column(db.Integer, nullable=False)
    surface = db.Column(db.String(15), nullable=False)
    temperature = db.Column(db.Integer, nullable=False)

with app.app_context():
    db.create_all()

runners_data = []
correlations_data = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global runners_data
    files = request.files.getlist("file")  # list of uploaded files
    try:
        all_races = []  # collect all races across all files
        runners_data = []  # collect all rows across all files
        runneridx = []
        for f in files:
            # Read CSV
            stream = io.StringIO(f.stream.read().decode("utf-8"))
            reader = csv.DictReader(stream)
            for row in reader:
                cleaned = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
                runners_data.append(cleaned)
                try:
                    race = Races(
                        name=cleaned['Athlete'],
                        event=cleaned['Event'],
                        date=cleaned['Date'],
                        distance=int(cleaned['Distance (m)']),
                        time_sec=parse_time_to_seconds(cleaned['Time']),
                        elevation=int(cleaned['Elevation Gain']),
                        humidity=int(cleaned['Humidity (%)']),
                        surface=cleaned['Surface'],
                        temperature=int(cleaned['Temperature (F)'])
                    )
                    # Check if the race already exists
                    if not Races.query.filter_by(
                            name=race.name,
                            event=race.event,
                            date=race.date,
                            distance=race.distance
                    ).first():
                        all_races.append(race)
                except Exception as e:
                    print(f"Skipping row due to error: {e}")
                    continue

        unique_names = []
        for i in range(len(runners_data)):
            if runners_data[i]['Athlete'] not in [u['Athlete'] for u in unique_names]:
                unique_names.append(runners_data[i])
                runneridx.append(i)

        # Save all races after processing all files
        if all_races:
            db.session.add_all(all_races)
            db.session.commit()
        flash(f"Successfully uploaded {len(runners_data)} records from {len(files)} file(s)!")
        return redirect(url_for('runner', idx=0))
    except Exception as e:
        print(f"Error processing file: {e}")
        flash(f"Error processing file: {e}")
        return redirect(url_for('index'))

@app.route('/runners', methods=['GET', 'POST'])
def runners_table():
    global runners_data
    if not runners_data:
        flash("No data uploaded yet.")
        return redirect(url_for('index'))
    # Filtration system
    filter = runners_data
    if request.method == 'POST':
        f = request.form.get('filter')
        distance = request.form.get('distance', '').strip()
        surface = request.form.get('surface', '').strip()
        if f != 'All Athletes':
            filter = [x for x in runners_data if x['Athlete'] == f]
        if distance != '':
            filter = [fd for fd in filter if str(fd['Distance (m)']) == distance]
        if surface != '':
            filter = [fd for fd in filter if str(fd['Surface']) == surface]
    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = 50  # rows per page
    start = (page - 1) * per_page
    end = start + per_page
    paginated_runners = filter[start:end]
    unique_names = []
    for i in runners_data:
        if i['Athlete'] not in [u['Athlete'] for u in unique_names]:
            unique_names.append(i)

    total_pages = (len(filter) + per_page - 1) // per_page
    names = sorted(list({r.get("Athlete", "").strip() for r in filter if "Athlete" in r}))

    return render_template(
        "secondary.html",
        unique_names=unique_names,
        runners=paginated_runners,
        names=names,
        page=page,
        total_pages=total_pages,

    )

@app.route('/db_table')
def db_table():
    data = Races.query.all()
    # Convert SQLAlchemy objects to dictionaries
    runners = []
    for r in data:
        runners.append({
            "ID": r.id,
            "Athlete": r.name,
            "Event": r.event,
            "Date": r.date,
            "Distance (m)": r.distance,
            "Time (s)": r.time_sec,
            "Elevation Gain": r.elevation,
            "Humidity (%)": r.humidity,
            "Surface": r.surface,
            "Temperature (F)": r.temperature
        })

    if not runners:
        flash("No data in database.")
        return redirect(url_for('index'))

    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = 50
    start = (page - 1) * per_page
    end = start + per_page
    paginated_runners = runners[start:end]
    total_pages = (len(runners) + per_page - 1) // per_page
    has_prev = page > 1
    has_next = page < total_pages

    return render_template(
        "db.html",
        runners=paginated_runners,
        page=page,
        total_pages=total_pages,
        has_prev=has_prev,
        has_next=has_next,

    )


@app.route('/runner/<int:idx>', methods=["GET", "POST"])
def runner(idx: int):
    global runners_data
    global correlations_data

    # Get the current runner's data first
    runner_row = runners_data[idx]
    current_athlete = runner_row.get("Athlete", "").strip()

    if request.method == "POST":
        event_name = request.form['selectEvent'].strip()
        # Filter by BOTH event name AND current athlete
        for j, race in enumerate(runners_data):
            if (race.get("Event", "").strip() == event_name and
                    race.get("Athlete", "").strip() == current_athlete):
                idx = j
                break
        # Update runner_row after potentially changing idx
        runner_row = runners_data[idx]

    athlete_name = runner_row.get("Athlete", "").strip()
    distance_m = to_int_safe(runner_row.get("Distance (m)"))
    if not athlete_name or distance_m is None:
        flash("Invalid runner data.")
        return redirect(url_for('runners_table'))

    # Rest of your code remains the same...
    raw_avg, ideal_avg, events = cumulative_averages(runners_data, athlete_name, distance_m)
    athlete_races = [r for r in runners_data if r.get("Athlete", "").strip() == athlete_name]
    df = pd.DataFrame(athlete_races)

    # Calculate correlations between data
    if not df.empty:
        df["Time (s)"] = df["Time"].apply(parse_time_to_seconds)
        df["Distance (m)"] = pd.to_numeric(df["Distance (m)"], errors="coerce")
        df["Temperature (F)"] = pd.to_numeric(df["Temperature (F)"], errors="coerce")
        df["Humidity (%)"] = pd.to_numeric(df["Humidity (%)"], errors="coerce")
        df["Elevation Gain"] = pd.to_numeric(df["Elevation Gain"], errors="coerce")
        df = df.dropna()

        correlations_data = round(df.corr(numeric_only=True)['Time (s)'], 2)
        correlations_data = correlations_data.drop("Time (s)", errors="ignore")
        correlations_data = correlations_data.drop("Distance (m)", errors="ignore")
        if not correlations_data.empty:
            # Get the highest correlation
            highest_feature = correlations_data.abs().idxmax()
            highest_value = correlations_data[highest_feature]
            percentage_strength = round(abs(highest_value) * 100, 2)
        else:
            highest_feature = None
            percentage_strength = None
    else:
        highest_feature = None
        percentage_strength = None

    # Gather past races for the athlete across all distances (for model)
    distances_m, times_sec = [], []
    runneridx = []
    race_names = []
    unique_names = []
    for i in range(len(runners_data)):
        if runners_data[i]['Athlete'] not in [u['Athlete'] for u in unique_names]:
            unique_names.append(runners_data[i])
            runneridx.append(i)

    # Filter race_names to only include races from the current athlete
    for race in runners_data:
        try:
            if race.get("Athlete", "").strip() != athlete_name:
                continue

            d = to_int_safe(race.get("Distance (m)"))
            name = race.get("Event", "").strip()
            if d is None or d <= 0:
                continue

            t = parse_time_to_seconds(race.get("Time", ""))
            distances_m.append(d)
            times_sec.append(t)
            race_names.append(name)
        except Exception:
            continue

    # Group times by distance
    times_by_distance = {}
    for d, t in zip(distances_m, times_sec):
        times_by_distance.setdefault(d, []).append(t)
    # Calculate SD per distance
    sd_by_distance = {d: np.std(times) for d, times in times_by_distance.items()}
    sd = round(sd_by_distance[distance_m], 3)
    if sd <= 15:
        c_score = "Very consistent with their other races"
    elif sd <= 30:
        c_score = "Consistent with their other races"
    else:
        c_score = "Inconsistent with their other races"
    prediction, target_distance = None, None
    if request.method == "POST":
        try:
            target_distance = to_int_safe(request.form.get("target_distance"))
            if target_distance and len(distances_m) >= 2:
                # Walrus syntax assigning the gascon_model to var model
                if (model := fit_gascon_model(np.array(distances_m), np.array(times_sec))):
                    pred_time_sec = predict_time(model, target_distance)
                    prediction = seconds_to_time(pred_time_sec) if pred_time_sec else None
            elif target_distance and len(distances_m) < 2:
                flash("Not enough data to build a prediction model (need at least 2 races).")
        except Exception as e:
            flash(f"Error making prediction: {e}")

    # Ideal adjustment for this specific race
    try:
        total_seconds = parse_time_to_seconds(runner_row.get("Time", ""))
        temp = to_float_safe(runner_row.get("Temperature (F)"), 60)
        h = to_float_safe(runner_row.get("Humidity (%)"), 60)
        e = to_float_safe(runner_row.get("Elevation Gain"), 0)
        args = (total_seconds, distance_m, temp, h, e)
        kwargs = dict(athlete=athlete_name, runners=runners_data)
        if len(times_by_distance) > 5:
            ideal_time_sec = normalize_time(*args, **kwargs)
        else:
            ideal_time_sec = heuristic(*args, **kwargs)
        ideal_time = seconds_to_time(ideal_time_sec) if ideal_time_sec is not None else None
        difference = round(total_seconds - ideal_time_sec, 2) if ideal_time_sec is not None else None
    except Exception:
        ideal_time = None
        difference = None
    # Calculate statistics for sidebar
    athlete_races = [r for r in runners_data if r.get("Athlete", "").strip() == athlete_name]
    total_races = len(athlete_races)
    if athlete_races:
        times = [parse_time_to_seconds(r.get("Time", "")) for r in athlete_races]
        times = [t for t in times if t > 0]
        best_time = seconds_to_time(min(times)) if times else None
        worst_time = seconds_to_time(max(times)) if times else None
    else:
        best_time = worst_time = None
    vo2max = vo2(distance_m, parse_time_to_seconds(runner_row.get("Time", "")))

    # Averages
    weights_series = weighted(runners_data, athlete_name, distance_m)
    min_len = min(len(events), len(weights_series))
    all_weights = weights_series[:min_len]
    weighted_avg = seconds_to_time(weights_series[-1]) if weights_series else None
    avg_time_sec = average_time_for_distance(runners_data, athlete_name, distance_m)
    average_display = seconds_to_time(avg_time_sec) if avg_time_sec else None
    avg_ideal_sec = average_ideal_for_distance(runners_data, athlete_name, distance_m)
    normal_display = seconds_to_time(avg_ideal_sec) if avg_ideal_sec else None
    race_time_sec = parse_time_to_seconds(runner_row.get("Time", ""))
    time_diff = avg_time_sec - race_time_sec
    sign = "+" if time_diff > 0 else "-"
    d = f"{sign}{seconds_to_time(abs(time_diff))}s"

    return render_template(
        "runner.html",
        runner=runner_row,
        ideal_time=ideal_time,
        difference=difference,
        prediction=prediction,
        all_weights=all_weights,
        weighted_avg=(f"{distance_m}m Average (Weighted): {weighted_avg}" if weighted_avg else "No data"),
        raw_avg=raw_avg,
        ideal_avg=ideal_avg,
        events=events,
        target_distance=target_distance,
        average=(f"{distance_m}m Average: {average_display}" if average_display else "No data"),
        normal_display=(f"{distance_m}m Ideal Conditions Average: {normal_display}" if normal_display else "No data"),
        total_races=total_races,
        best_time=best_time,
        worst_time=worst_time,
        d=d,
        sd=sd,
        c_score=c_score,
        percentage_strength=percentage_strength,
        highest_feature=highest_feature,
        vo2max=vo2max,
        unique_names=unique_names,
        race_names=race_names,
        runneridx=runneridx
    )


if __name__ == '__main__':
    debug_mode = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=True, host="0.0.0.0")

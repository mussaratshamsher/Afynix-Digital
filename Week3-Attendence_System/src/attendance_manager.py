"""Attendance management module with statistics and export options."""

import csv
import os
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd


class AttendanceManager:
    """Manages attendance records with enhanced features."""

    def __init__(self, attendance_file: Optional[str] = None):
        """Initialize attendance manager.

        Args:
            attendance_file: Path to attendance CSV file
        """
        if attendance_file is None:
            from pathlib import Path
            base_dir = Path(__file__).parent.parent
            attendance_file = base_dir / "data" / "attendance" / "attendance.csv"

        self.attendance_file = Path(attendance_file)
        self.attendance_file.parent.mkdir(parents=True, exist_ok=True)

        # Track who has marked attendance today
        self._today_attendance: set = set()
        self._load_today_attendance()

    def _load_today_attendance(self):
        """Load today's attendance to prevent duplicates."""
        if not self.attendance_file.exists():
            return

        today = date.today().isoformat()

        try:
            with open(self.attendance_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('Date') == today:
                        self._today_attendance.add(row.get('Name', ''))
        except Exception as e:
            print(f"Error loading attendance: {e}")

    def mark_attendance(self, name: str) -> bool:
        """Mark attendance for a person.

        Args:
            name: Name of the person

        Returns:
            True if marked, False if already marked today
        """
        # Check if already marked today
        if name in self._today_attendance:
            return False

        # Get current timestamp
        now = datetime.now()
        today_date = now.strftime('%Y-%m-%d')
        current_time = now.strftime('%H:%M:%S')

        # Create file if not exists
        file_exists = self.attendance_file.exists()

        # Write attendance record
        with open(self.attendance_file, 'a', newline='') as f:
            fieldnames = ['Name', 'Date', 'Time']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                'Name': name,
                'Date': today_date,
                'Time': current_time
            })

        # Update today's attendance
        self._today_attendance.add(name)

        return True

    def is_marked_today(self, name: str) -> bool:
        """Check if person has marked attendance today.

        Args:
            name: Name of the person

        Returns:
            True if marked today, False otherwise
        """
        return name in self._today_attendance

    def get_today_attendance(self) -> List[Dict]:
        """Get today's attendance records.

        Returns:
            List of dicts with Name, Date, Time
        """
        today = date.today().isoformat()
        return self.get_attendance_by_date(today)

    def get_attendance_by_date(self, target_date: str) -> List[Dict]:
        """Get attendance records for a specific date.

        Args:
            target_date: Date in YYYY-MM-DD format

        Returns:
            List of attendance records
        """
        records = []

        if not self.attendance_file.exists():
            return records

        try:
            with open(self.attendance_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('Date') == target_date:
                        records.append({
                            'Name': row.get('Name'),
                            'Date': row.get('Date'),
                            'Time': row.get('Time')
                        })
        except Exception as e:
            print(f"Error reading attendance: {e}")

        return records

    def get_attendance_count_today(self) -> int:
        """Get count of attendance for today.

        Returns:
            Number of people marked present today
        """
        return len(self._today_attendance)

    def clear_today(self):
        """Clear today's attendance tracking (for new day)."""
        self._today_attendance.clear()

    def remove_attendance(self, name: str, target_date: Optional[str] = None) -> bool:
        """Remove attendance record for a person.

        Args:
            name: Name of person
            target_date: Date in YYYY-MM-DD format (default: today)

        Returns:
            True if removed, False if not found
        """
        if target_date is None:
            target_date = date.today().isoformat()

        if not self.attendance_file.exists():
            return False

        # Read all records
        records = []
        removed = False

        try:
            with open(self.attendance_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('Name') == name and row.get('Date') == target_date:
                        removed = True
                        continue
                    records.append(row)
        except Exception as e:
            print(f"Error reading attendance: {e}")
            return False

        # Write back
        try:
            with open(self.attendance_file, 'w', newline='') as f:
                fieldnames = ['Name', 'Date', 'Time']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)

            # Update tracking
            if target_date == date.today().isoformat():
                self._today_attendance.discard(name)

        except Exception as e:
            print(f"Error writing attendance: {e}")
            return False

        return removed

    def get_all_attendance(self) -> List[Dict]:
        """Get all attendance records.

        Returns:
            List of all attendance records
        """
        if not self.attendance_file.exists():
            return []

        records = []
        try:
            with open(self.attendance_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append({
                        'Name': row.get('Name'),
                        'Date': row.get('Date'),
                        'Time': row.get('Time')
                    })
        except Exception as e:
            print(f"Error reading attendance: {e}")

        return records

    # Enhanced statistics methods
    def get_statistics(self, days: int = 7) -> Dict:
        """Get attendance statistics.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with statistics
        """
        all_records = self.get_all_attendance()

        if not all_records:
            return {
                "total_attendance": 0,
                "unique_persons": 0,
                "daily_average": 0,
                "most_regular": [],
                "last_7_days": {},
                "first_arrival": None,
                "late_arrivals": []
            }

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(all_records)

        # Get last N days
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        dates_range = [(start_date + timedelta(days=i)).isoformat() for i in range(days + 1)]

        # Filter to date range
        df_week = df[df['Date'].isin(dates_range)]

        # Calculate statistics
        stats = {
            "total_attendance": len(df_week),
            "unique_persons": df_week['Name'].nunique() if not df_week.empty else 0,
            "daily_average": round(len(df_week) / days, 1) if days > 0 else 0,
            "last_7_days": df_week['Date'].value_counts().to_dict() if not df_week.empty else {},
            "most_regular": self._get_most_regular(df_week, limit=5),
            "first_arrival": self._get_first_arrival(df_week),
            "late_arrivals": self._get_late_arrivals(df_week, threshold="09:00:00")
        }

        return stats

    def _get_most_regular(self, df: pd.DataFrame, limit: int = 5) -> List[Dict]:
        """Get most regular attendees."""
        if df.empty:
            return []

        counts = df['Name'].value_counts().head(limit)
        return [
            {"name": name, "count": int(count)}
            for name, count in counts.items()
        ]

    def _get_first_arrival(self, df: pd.DataFrame) -> Optional[Dict]:
        """Get first arrival of the period."""
        if df.empty or df[df['Date'] == date.today().isoformat()].empty:
            return None

        today_df = df[df['Date'] == date.today().isoformat()]
        first = today_df.loc[today_df['Time'].idxmin()]

        return {
            "name": first['Name'],
            "time": first['Time']
        }

    def _get_late_arrivals(self, df: pd.DataFrame, threshold: str = "09:00:00") -> List[Dict]:
        """Get late arrivals after threshold."""
        if df.empty:
            return []

        late = df[df['Time'] > threshold]
        return [
            {"name": row['Name'], "time": row['Time']}
            for _, row in late.iterrows() if row['Date'] == date.today().isoformat()
        ]

    def get_person_history(self, name: str) -> List[Dict]:
        """Get attendance history for a specific person.

        Args:
            name: Name of person

        Returns:
            List of attendance records
        """
        all_records = self.get_all_attendance()
        return [r for r in all_records if r['Name'] == name]

    def get_person_statistics(self, name: str) -> Dict:
        """Get statistics for a specific person.

        Args:
            name: Name of person

        Returns:
            Dictionary with person statistics
        """
        history = self.get_person_history(name)

        if not history:
            return {
                "name": name,
                "total_days": 0,
                "first_seen": None,
                "last_seen": None,
                "streak": 0,
                "average_time": None
            }

        dates = [r['Date'] for r in history]
        times = [r['Time'] for r in history if r['Date'] == date.today().isoformat()]

        # Calculate streak
        streak = 0
        check_date = date.today()
        sorted_dates = sorted(set(dates), reverse=True)

        for d in sorted_dates:
            if (date.fromisoformat(d) - check_date).days <= 1:
                streak += 1
                check_date = date.fromisoformat(d)
            else:
                break

        # Average arrival time
        avg_time = None
        if times:
            total_seconds = sum(
                int(t.split(':')[0]) * 3600 +
                int(t.split(':')[1]) * 60 + int(t.split(':')[2])
            for t in times
            )
            avg_seconds = total_seconds // len(times)
            avg_time = f"{avg_seconds // 3600:02d}:{ (avg_seconds % 3600) // 60:02d}:{avg_seconds % 60:02d}"

        return {
            "name": name,
            "total_days": len(set(dates)),
            "first_seen": min(dates),
            "last_seen": max(dates),
            "streak": streak,
            "average_time": avg_time
        }

    def generate_report(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate attendance report for date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with attendance summary
        """
        all_records = self.get_all_attendance()

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)

        # Filter to date range
        if not df.empty and 'Date' in df.columns:
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

        if df.empty:
            return df

        # Group by person and date
        summary = df.groupby(['Name', 'Date']).size().reset_index(name='Count')
        pivot = summary.pivot(index='Name', columns='Date', values='Count').fillna(0)

        return pivot

    # Export methods
    def export_csv(self, output_file: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None) -> str:
        """Export attendance to CSV file.

        Args:
            output_file: Output file path (default: use default file)
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Path to exported file
        """
        if output_file is None:
            output_file = self.attendance_file

        if start_date and end_date:
            # Export date range
            all_records = self.get_all_attendance()
            df = pd.DataFrame(all_records)
            if not df.empty and 'Date' in df.columns:
                df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            df.to_csv(output_file, index=False)
        else:
            # Copy file
            import shutil
            shutil.copy(self.attendance_file, output_file)

        return str(output_file)

    def export_json(self, output_file: Optional[str] = None,
                  include_stats: bool = False) -> str:
        """Export attendance to JSON file.

        Args:
            output_file: Output file path
            include_stats: Include statistics

        Returns:
            Path to exported file
        """
        if output_file is None:
            output_file = self.attendance_file.with_suffix('.json')

        data = {
            "attendance": self.get_all_attendance(),
            "exported_at": datetime.now().isoformat()
        }

        if include_stats:
            data["statistics"] = self.get_statistics()

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        return str(output_file)

    def export_excel(self, output_file: Optional[str] = None) -> str:
        """Export attendance to Excel file.

        Args:
            output_file: Output file path

        Returns:
            Path to exported file
        """
        if output_file is None:
            output_file = self.attendance_file.with_suffix('.xlsx')

        all_records = self.get_all_attendance()
        df = pd.DataFrame(all_records)

        # Write to Excel
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Attendance', index=False)

            # Add statistics sheet
            stats = self.get_statistics()
            stats_df = pd.DataFrame([stats])
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)

        return str(output_file)

    def import_csv(self, input_file: str) -> int:
        """Import attendance from CSV file.

        Args:
            input_file: Path to CSV file

        Returns:
            Number of records imported
        """
        if not Path(input_file).exists():
            return 0

        count = 0
        try:
            with open(input_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get('Name')
                    record_date = row.get('Date')
                    record_time = row.get('Time')

                    if name and record_date:
                        # Append to file
                        with open(self.attendance_file, 'a', newline='') as out:
                            writer = csv.DictWriter(out, fieldnames=['Name', 'Date', 'Time'])
                            if self.attendance_file.stat().st_size == 0:
                                writer.writeheader()
                            writer.writerow({
                                'Name': name,
                                'Date': record_date,
                                'Time': record_time
                            })
                        count += 1
        except Exception as e:
            print(f"Error importing: {e}")

        return count

    def clear_all(self):
        """Clear all attendance records."""
        if self.attendance_file.exists():
            self.attendance_file.unlink()
        self._today_attendance.clear()
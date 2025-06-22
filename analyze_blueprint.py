#!/usr/bin/env python3
"""
Analyze street distribution in blueprint parquet files using DuckDB.

Usage:
    python analyze_blueprint.py [parquet_file]
    
If no file is specified, it will look for files matching the pattern:
    pgcopy/blueprint*.parquet
"""

import sys
import glob
import os

# Try to import duckdb, provide installation instructions if not available
try:
    import duckdb
except ImportError:
    print("DuckDB not found. Please install it with:")
    print("  pip install duckdb")
    sys.exit(1)

def analyze_blueprint(parquet_file):
    """Analyze the street distribution in a blueprint parquet file."""
    
    if not os.path.exists(parquet_file):
        print(f"Error: File '{parquet_file}' not found.")
        return
    
    print(f"\nAnalyzing: {parquet_file}")
    print("=" * 80)
    
    # Connect to DuckDB
    conn = duckdb.connect(':memory:')
    
    # Main analysis query
    query = f"""
    WITH street_breakdown AS (
        SELECT 
            (present >> 56) AS street_id,
            CASE (present >> 56)
                WHEN 0 THEN 'Preflop'
                WHEN 1 THEN 'Flop'
                WHEN 2 THEN 'Turn'
                WHEN 3 THEN 'River'
                ELSE 'Unknown'
            END AS street,
            COUNT(*) as edge_count,
            COUNT(DISTINCT CONCAT(CAST(history AS VARCHAR), '-', CAST(present AS VARCHAR), '-', CAST(futures AS VARCHAR))) as infoset_count,
            COUNT(DISTINCT present) as unique_abstractions,
            MIN(present & 4095) as min_cluster,  -- 0xFFF = 4095
            MAX(present & 4095) as max_cluster   -- 0xFFF = 4095
        FROM '{parquet_file}'
        GROUP BY street_id
    ),
    totals AS (
        SELECT 
            SUM(edge_count) as total_edges,
            SUM(infoset_count) as total_infosets
        FROM street_breakdown
    )
    SELECT 
        s.street,
        s.edge_count,
        s.infoset_count,
        s.unique_abstractions,
        s.min_cluster,
        s.max_cluster,
        ROUND(100.0 * s.edge_count / t.total_edges, 2) AS edge_percentage,
        ROUND(100.0 * s.infoset_count / t.total_infosets, 2) AS infoset_percentage
    FROM street_breakdown s, totals t
    ORDER BY s.street_id;
    """
    
    try:
        # Execute the query
        result = conn.execute(query).fetchall()
        
        # Print results in a nice table format
        print(f"{'Street':<10} {'Edges':>12} {'Infosets':>12} {'Unique Abs':>12} {'Clusters':>15} {'Edge %':>8} {'Info %':>8}")
        print("-" * 80)
        
        for row in result:
            street, edges, infosets, unique_abs, min_cluster, max_cluster, edge_pct, info_pct = row
            cluster_range = f"{min_cluster}-{max_cluster}" if min_cluster is not None else "N/A"
            print(f"{street:<10} {edges:>12,} {infosets:>12,} {unique_abs:>12,} {cluster_range:>15} {edge_pct:>7.1f}% {info_pct:>7.1f}%")
        
        # Get summary statistics
        summary_query = f"""
        SELECT 
            COUNT(*) as total_edges,
            COUNT(DISTINCT CONCAT(CAST(history AS VARCHAR), '-', CAST(present AS VARCHAR), '-', CAST(futures AS VARCHAR))) as unique_infosets,
            COUNT(DISTINCT present) as unique_abstractions
        FROM '{parquet_file}';
        """
        
        summary = conn.execute(summary_query).fetchone()
        total_edges, unique_infosets, unique_abstractions = summary
        
        print("\n" + "=" * 80)
        print(f"Total edges:          {total_edges:,}")
        print(f"Unique infosets:      {unique_infosets:,}")
        print(f"Unique abstractions:  {unique_abstractions:,}")
        
        # Sample some abstractions to verify encoding
        print("\n" + "=" * 80)
        print("Sample abstractions (first 10):")
        print(f"{'Present':>18} {'Street ID':>10} {'Street':>10} {'Cluster':>10} {'Hex':>18}")
        print("-" * 80)
        
        sample_query = f"""
        SELECT 
            present,
            (present >> 56) AS street_id,
            CASE (present >> 56)
                WHEN 0 THEN 'Preflop'
                WHEN 1 THEN 'Flop'
                WHEN 2 THEN 'Turn'
                WHEN 3 THEN 'River'
                ELSE 'Unknown'
            END as street_name,
            (present & 4095) AS cluster_index,  -- 0xFFF = 4095
            printf('0x%016X', present) as present_hex
        FROM '{parquet_file}'
        GROUP BY present
        ORDER BY present
        LIMIT 10;
        """
        
        samples = conn.execute(sample_query).fetchall()
        for present, street_id, street_name, cluster_index, present_hex in samples:
            print(f"{present:>18} {street_id:>10} {street_name:>10} {cluster_index:>10} {present_hex:>18}")
            
    except Exception as e:
        print(f"Error executing query: {e}")
    finally:
        conn.close()

def main():
    if len(sys.argv) > 1:
        # Use specified file
        parquet_file = sys.argv[1]
    else:
        # Look for blueprint parquet files
        parquet_files = glob.glob("pgcopy/blueprint*.parquet")
        
        if not parquet_files:
            print("No blueprint parquet files found in pgcopy/ directory.")
            print("Please specify a file path as an argument.")
            sys.exit(1)
        
        if len(parquet_files) == 1:
            parquet_file = parquet_files[0]
        else:
            print("Multiple blueprint files found:")
            for i, f in enumerate(parquet_files):
                print(f"  {i+1}. {f}")
            
            try:
                choice = int(input("\nSelect file number (or 0 to analyze all): "))
                if choice == 0:
                    for f in parquet_files:
                        analyze_blueprint(f)
                    return
                elif 1 <= choice <= len(parquet_files):
                    parquet_file = parquet_files[choice - 1]
                else:
                    print("Invalid choice.")
                    sys.exit(1)
            except (ValueError, KeyboardInterrupt):
                print("\nCancelled.")
                sys.exit(1)
    
    analyze_blueprint(parquet_file)

if __name__ == "__main__":
    main() 
import { FC, useEffect, useRef } from "react";
import { Chart, ChartItem, registerables } from "chart.js";

Chart.register(...registerables);
let isChartsInited = false;

export const Home: FC = () => {
  const streamImageRef = useRef<HTMLImageElement | null>(null);

  const chart1Ref = useRef<HTMLCanvasElement | null>(null);
  const chart2Ref = useRef<HTMLCanvasElement | null>(null);
  const chart3Ref = useRef<HTMLCanvasElement | null>(null);
  const chart4Ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!isChartsInited && chart1Ref.current) {
      isChartsInited = true;

      const ctx = chart1Ref.current.getContext("2d") as ChartItem;
      const socket = new WebSocket("ws://localhost:80/ws");

      const chart1 = new Chart(ctx, {
        type: "line",
        data: {
          labels: Array(100)
            .fill(null)
            .map(() => ""),
          datasets: [
            {
              label: "# of Votes",
              data: [0],
              backgroundColor: "#ffd736",
              borderColor: "#ffd736",
              borderWidth: 1,
              pointRadius: 2,
            },
          ],
        },
        options: {
          animation: {
            duration: 0,
          },
          scales: {
            y: {
              beginAtZero: true,
            },
            x: {
              max: 100,
            },
          },
        },
      });

      setInterval(() => {
        chart1.data.datasets[0].data.push((Math.random() * 50) | 0);

        if (chart1.data.datasets[0].data.length > 100) {
          chart1.data.datasets[0].data = chart1.data.datasets[0].data.slice(1);
        }

        chart1.update();
      }, 100);

      socket.onmessage = (event) => {
        const image = event.data;

        if (streamImageRef.current) {
          streamImageRef.current.src = `data:image/png;base64,${image}`;
        }
      };
    }
  }, []);

  return (
    <div className="page">
      <div className="left-items">
        <canvas className="chart1" ref={chart1Ref}></canvas>
        <div className="chartsContainer">
          <canvas className="chart2" ref={chart2Ref}></canvas>
          <canvas className="chart3" ref={chart3Ref}></canvas>
        </div>
        <canvas className="chart4" ref={chart4Ref}></canvas>
      </div>
      <div className="right-items">
        <img ref={streamImageRef} src="" alt="" className="stream" />
      </div>
    </div>
  );
};

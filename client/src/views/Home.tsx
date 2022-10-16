import { FC, useEffect, useRef } from "react";
import { Chart, ChartItem, registerables } from "chart.js";
import { WarningOutlined } from "@ant-design/icons";

Chart.register(...registerables);
let isChartsInited = false;

export const Home: FC = () => {
  const streamImageRef = useRef<HTMLImageElement | null>(null);

  const chart1Ref = useRef<HTMLCanvasElement | null>(null);
  const chart2Ref = useRef<HTMLCanvasElement | null>(null);
  const chart3Ref = useRef<HTMLCanvasElement | null>(null);
  const chart4Ref = useRef<HTMLCanvasElement | null>(null);

  const emptyLineWarningRef = useRef<HTMLHeadingElement | null>(null);
  const negabaritWarningRef = useRef<HTMLHeadingElement | null>(null);

  useEffect(() => {
    if (
      !isChartsInited &&
      chart1Ref.current &&
      chart2Ref.current &&
      chart3Ref.current &&
      chart4Ref.current
    ) {
      isChartsInited = true;

      const socket = new WebSocket("ws://localhost:80/ws");

      const chart1 = new Chart(
        chart1Ref.current.getContext("2d") as ChartItem,
        {
          type: "line",
          data: {
            labels: Array(100)
              .fill(null)
              .map(() => ""),
            datasets: [
              {
                label: "Средний размер объектов",
                data: [],
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
        }
      );

      const chart2 = new Chart(
        chart2Ref.current.getContext("2d") as ChartItem,
        {
          type: "bar",
          data: {
            labels: ["Соотношение больших кусков, %"],
            datasets: [
              {
                label: "Большие куски",
                data: [0],
                backgroundColor: "#33a3ff",
              },
            ],
          },
        }
      );

      const chart3 = new Chart(
        chart3Ref.current.getContext("2d") as ChartItem,
        {
          type: "pie",
          data: {
            labels: ["1", "2", "3", "4", "5", "6", "7"],
            datasets: [
              {
                label: "Процентное соотношение классов",
                data: [0, 0, 0, 0, 0, 0, 0],
                backgroundColor: [
                  "#ff8f33",
                  "#c2ff33",
                  "#36ff33",
                  "#33a3ff",
                  "#9933ff",
                  "#ff337a",
                  "#3352ff",
                ],
              },
            ],
          },
        }
      );

      const chart4 = new Chart(
        chart4Ref.current.getContext("2d") as ChartItem,
        {
          type: "line",
          data: {
            labels: Array(100)
              .fill(null)
              .map(() => ""),
            datasets: [
              {
                label: "Максимальные по размеру объекты",
                data: [],
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
        }
      );

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === "image") {
          if (streamImageRef.current) {
            streamImageRef.current.src = `data:image/png;base64,${data.bytes}`;
          }

          if (data.negabarit_found) {
            negabaritWarningRef.current?.classList.add("warning--active");
          } else {
            negabaritWarningRef.current?.classList.remove("warning--active");
          }

          if (data.empty_line) {
            emptyLineWarningRef.current?.classList.add("warning--active");
          } else {
            emptyLineWarningRef.current?.classList.remove("warning--active");
          }
        }

        if (data.type === "metrics") {
          data.histogram = JSON.parse(data.histogram);

          chart1.data.datasets[0].data.push(
            data.average_predicted_size as never
          );

          if (chart1.data.datasets[0].data.length > 100) {
            chart1.data.datasets[0].data =
              chart1.data.datasets[0].data.slice(1);
          }

          chart1.update();

          chart2.data.datasets[0].data = [
            data.average_large_particle_percentage * 100,
          ];
          chart2.update();

          chart3.data.datasets[0].data = Object.values(data.histogram);
          chart3.update();

          chart4.data.datasets[0].data.push(data.max_predicted_size as never);

          if (chart4.data.datasets[0].data.length > 100) {
            chart4.data.datasets[0].data =
              chart4.data.datasets[0].data.slice(1);
          }

          chart4.update();
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
          <canvas className="chart4" ref={chart4Ref}></canvas>
        </div>
        <canvas className="chart3" ref={chart3Ref}></canvas>
      </div>
      <div className="right-items">
        <img ref={streamImageRef} src="" alt="" className="stream" />
        <div className="warnings">
          <h3 className="warning empty-line" ref={emptyLineWarningRef}>
            <WarningOutlined /> Линия пуста
          </h3>
          <h3 className="warning negabarit-found" ref={negabaritWarningRef}>
            <WarningOutlined /> Найден негабаритный объект
          </h3>
        </div>
      </div>
    </div>
  );
};

/* ECDR Component: MethodMatrixComponent.js
 * Interactive Method Comparison Matrix (M00–M16) across 5,053 Streaming Windows.
 */
(function (global) {
  'use strict';
  global.ECDR_COMPONENTS = global.ECDR_COMPONENTS || {};

  function render() {
    const rows = [
      ['M00', 'Frozen full core', 'baseline', '0', '1.0000', '89.05%', '0.7763', '0.342 ms', '1.00×', '0.0%', 'OPERATIONAL_REFERENCE'],
      ['M01', 'Full wrapper', 'wrapper', '0', '1.0000', '89.05%', '0.7763', '1.677 ms', '0.20×', '-390.2%', 'OPERATIONAL_REFERENCE'],
      ['M02', 'Exact input replay', 'singleton', '0', '1.0000', '89.05%', '0.7763', '0.479 ms', '0.71×', '-40.0%', 'PASS'],
      ['M03', 'Exact incremental computation', 'feature_cache', '5,034', '1.0000', '89.05%', '0.7763', '1.927 ms', '0.18×', '-463.5%', 'PASS'],
      ['M04', 'Heuristic reuse', 'heuristic', '0', '1.0000', '89.05%', '0.7763', '0.497 ms', '0.69×', '-45.4%', 'FAIL'],
      ['M05', 'Global Lipschitz cache', 'certification', '0', '1.0000', '89.05%', '0.7763', '0.509 ms', '0.67×', '-48.9%', 'PASS'],
      ['M06', 'IBP diagonal box', 'certification', '0', '1.0000', '89.05%', '0.7763', '7.128 ms', '0.05×', '-1984.0%', 'PASS'],
      ['M07', 'IBP fixed-overlap wave', 'certification', '0', '1.0000', '89.05%', '0.7763', '6.564 ms', '0.05×', '-1819.2%', 'PASS'],
      ['M08', 'Correlated candidate hull', 'ablation', '0', '1.0000', '89.05%', '0.7763', '0.463 ms', '0.74×', '-35.3%', 'PASS'],
      ['M09', 'Native correlated bounds', 'certification', '0', '1.0000', '89.05%', '0.7763', '0.448 ms', '0.76×', '-31.0%', 'PASS'],
      ['M10', 'CROWN diagonal', 'certification', '0', '1.0000', '89.05%', '0.7763', '0.450 ms', '0.76×', '-31.6%', 'PASS'],
      ['M11', 'CROWN wave/correlated', 'certification', '0', '1.0000', '89.05%', '0.7763', '0.448 ms', '0.76×', '-31.0%', 'PASS'],
      ['M12', 'Fixed-anchor policy control', 'policy_axis', '0', '1.0000', '89.05%', '0.7763', '0.497 ms', '0.69×', '-45.2%', 'PASS'],
      ['M13', 'Finite-horizon policy control', 'policy_axis', '0', '1.0000', '89.05%', '0.7763', '6.146 ms', '0.06×', '-1696.8%', 'PASS'],
      ['M14', 'LipCache comparator', 'comparator', '0', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'AUDITED_UNSUPPORTED'],
      ['M15', 'Prior-work audit', 'audit_only', '0', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'AUDITED_UNSUPPORTED'],
      ['M16', 'OWNED_REFERENCE baseline', 'secondary', '0', '1.0000', '89.05%', '0.7763', '0.467 ms', '0.73×', '-36.7%', 'PASS']
    ];

    const headers = ['ID', 'Tên phương pháp', 'Nhóm kỹ thuật', 'Số lần tái sử dụng', 'Độ bảo toàn (A_full)', 'Accuracy', 'Macro-F1', 'Độ trễ trung bình', 'Tốc độ vs M00', 'Tiết kiệm ròng', 'Bảo toàn an toàn'];

    return `
      <div class="component-card method-matrix-component">
        <div class="component-header">
          <span class="component-badge">KHÔNG GIAN THỰC NGHIỆM</span>
          <h3 class="component-title">Ma trận đối sánh toàn diện 17 phương pháp (M00–M16)</h3>
          <p class="component-desc">Đo đạc đồng bộ trên toàn bộ chuỗi test stream 5,053 cửa sổ (5,034 cặp kề cận trong cùng bản ghi).</p>
        </div>
        <div class="table-wrap">
          <table class="data-table">
            <thead>
              <tr>${headers.map(h => `<th scope="col">${h}</th>`).join('')}</tr>
            </thead>
            <tbody>
              ${rows.map(r => `
                <tr>
                  <td><strong>${r[0]}</strong></td>
                  <td>${r[1]}</td>
                  <td><span class="badge badge-neutral">${r[2]}</span></td>
                  <td><strong>${r[3]}</strong></td>
                  <td>${r[4]}</td>
                  <td>${r[5]}</td>
                  <td>${r[6]}</td>
                  <td>${r[7]}</td>
                  <td><strong>${r[8]}</strong></td>
                  <td style="color:${r[9].startsWith('-') ? '#dc2626' : '#16a34a'}">${r[9]}</td>
                  <td>
                    ${r[10] === 'PASS' ? '<span class="badge badge-success">PASS</span>' :
                      r[10] === 'FAIL' ? '<span class="badge badge-danger">FAIL</span>' :
                      `<span class="badge badge-neutral">${r[10]}</span>`}
                  </td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  global.ECDR_COMPONENTS.MethodMatrixComponent = { render };
})(typeof window !== 'undefined' ? window : globalThis);

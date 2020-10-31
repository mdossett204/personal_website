$(function() {
	var numCols = {"mileage":350000, "mpg":475, "engineSize":6.8};
	
	function createTable(result){
		let $table = $("<table/>");
		$.each(result, function(colName, colValues){
			createRow($table, colName, colValues);	
		});
		$.each(numCols, function(colName, _){
			createRow($table, colName);
		});
		let $button = $("<button/>").text("Get Price")
									.attr({"id":"getPrice", "style":"margin-top:10px;"});
		let $col = $("<td/>").attr({"colspan": 2, "style": "text-align:center;"});
		let $row = $("<tr/>").append($col);
		$col.append($button);
		$table.append($row);
		return $table;
	}
	
	function createRow(table, colName, colValues){
		let $row = $("<tr/>");
		createLabelCol($row, colName);
		if (typeof colValues === "undefined"){
			createNumColInput($row, colName);
		} else {
			createOptionCol($row, colName, colValues);
		}
		table.append($row);
	}
	
	function createLabelCol(row, colName){
		let $col = $("<td/>").text(colName.toUpperCase());
		row.append($col);
	}
	
	function createOptionCol(row, colName, colValues){
		let $col = $("<td/>");
		let $input = $("<input/>").attr({"list": colName+"s", "name":colName, "id":colName});
		$col.append($input);
		let $dataList = $("<datalist/>").attr("id", colName+"s");
		createOptions($dataList, colValues);
		$col.append($dataList);
		row.append($col);
	}
	
	function createNumColInput(row,colName){
		let $col = $("<td/>");
		let $input = $("<input/>").attr({type:"text", "name":colName, "id":colName});
		$col.append($input);
		row.append($col);
	}
	
	function createOptions(dataList,options){
		options.forEach(function(option){
			let	$option = $("<option/>").attr({"value":option});
			dataList.append($option);
		});
	}
	
	$.ajax({
  		dataType: "json",
  		url: "/assets/js/feature_options.json",
  		success: function(result){
  			$("#carData").append(createTable(result));
  		}
	});
	
	function validateInput(name, input){
		if (name in numCols){
			if (input.match(/^\d+\.?\d*$/)){
				let upperLimit = numCols[name];
				let output = parseFloat(input);
				if (output > upperLimit){
					alert("Your input for "+ name.toUpperCase() +" is too high.");
					return false
				}
				return output;
			} 
			alert("The input for "+ name.toUpperCase() + " needs to be a number.");
			return false;
		}
		if (name === "year"){
			return parseFloat(input);
		} 
		return input;
	};
	
	$("#dialog").dialog({
		autoOpen : false, modal : true, show : "blind", hide : "blind",
		close: function(){$("#dialog").empty().hide();}})

	
	$("#carData").on("click", "#getPrice", function(){
		var data = {};
		var success = true;
		$("#carData input").each(function(){
			let name = $(this).attr("name");
			let value = validateInput(name, $(this).val());
			if (value === false){
				success = false;
				return false;
			}
			data[name] = value;
		});
		if (!success){
			return;
		}
		$.ajax({
			type: "POST",
			url: "http://localhost:8081/carPrice",
			dataType: "json",
			data: JSON.stringify(data),
			contentType: "application/json",
			success: function(result){
				// alert("The predicted price for your car is "+ result.pred_price);
				let message = $("<p/>").text("The predicted price for your car is "
											 + result.pred_price);
				$("#dialog").append(message);
				$("#dialog").dialog("open");
			},
			error: function(){
				let message = $("<p/>").text("Sorry we are experiencing difficulties.");
				$("#dialog").append(message);
				$("#dialog").dialog("open");
			}
		});
	});
});

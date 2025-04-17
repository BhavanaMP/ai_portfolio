import { Injectable } from '@angular/core';
import { HttpClient, HttpResponse,HttpHeaders} from '@angular/common/http';
import { Observable, of,throwError } from 'rxjs';
import { Result } from '../model/results.model';
import { environment } from '../../environments/environment';
import { map, catchError } from 'rxjs/operators';
import { ReRank } from '../model/rerank.model';
///import { AnyAaaaRecord } from 'dns';

@Injectable({
  providedIn: 'root'
})
export class ServletService {
  name: any;
  query:any;
  private backendUrl: string=environment.apiUrl;
  private searchUrl:string=environment.apiUrl+"/api/getQuery";
  private resetUrl:string=environment.apiUrl+"/api/logout?reset=";
  private rerankUrl:string=environment.apiUrl+"/api/reRankData";
  results: Result[] = [];
  reRankobj: ReRank[] = [];
  constructor(private http: HttpClient) { }

  setData(data: any) {
    this.name=data;
    console.log(this.name);
  }
  getQuery(){
    return this.query;
  }
  getobj(results:Result[]){
    this.results=results;

  }

  sendQuery(query: any): Observable<any> {
    console.log("Requesting the backend for results to a query"+this.searchUrl);
    this.query=query;
    return this.http.post<any>(this.searchUrl, query);
  }
  reset(reset:any):Observable<any> {
    
    console.log(reset); //string
    console.log(this.resetUrl+"\""+reset+"\"");
   //return this.http.get<any>(this.resetUrl+"\""+reset+"\"");
   return this.http.post<any>(this.resetUrl, reset);

  }
  
  reRank(reRankobj:any): Observable<any> {
    //pass object as parameter to send ratings and result list number to backend
    console.log("Inside Rerank service"+reRankobj);
    this.reRankobj = reRankobj;
    this.reRankobj.forEach((element) => {
      console.log(element.ratings)
    });
   return this.http.post<any>(this.rerankUrl, reRankobj);
  }

  /*getResults():Observable<Result[]> {
    console.log(this.backendUrl);
    return this.http.get<Result[]>(this.backendUrl).pipe(
      map((data: Users[]) => {
        return data;
      }), catchError( error => {
        return throwError( 'Something went wrong!' );
      })
   );
  }*/
  getResults():Observable<any> {
    //return this.http.get(this.backendUrl);
    return this.http.get<Result[]>(this.backendUrl).
        pipe(
           map((data: Result[]) => {
             console.log(data);
             return data;
           }), catchError( error => {
             return throwError( 'Something went wrong!' );
           })
        )

  }

}
